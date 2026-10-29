#!/bin/bash
# run_all_st.sh - 一键执行所有ST用例
#
# 用法:
#   bash run_all_st.sh [日志目录] [选项]
#
# 日志目录：每个用例一个子目录（用例名去掉 .sh），内含 stdout.log / stderr.log；
#           可用位置参数或 --log-dir 指定，不传则不记录日志
#
# 选项:
#   --log-dir 目录       日志保存目录（等价于位置参数）
#   --case 用例名        只执行指定用例，可重复传入；名字带不带 .sh 后缀均可
#   --timeout 秒数       单用例超时秒数，超时杀掉整个进程组；0=不限时（默认）
#   --result-json 文件   逐用例结果写入该 JSON 文件，供 CI/脚本消费
#   --list               只列出 test_cases.txt 中已启用的用例，不执行
#   -h, --help           显示本帮助
#
# 退出码: 0=全部通过  1=有用例失败或超时  2=用法或配置错误
#
# 用例在 test_cases.txt 中注册，每行一个，按文件顺序执行

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CASE_FILE="${SCRIPT_DIR}/test_cases.txt"
TEST_DIR="${SCRIPT_DIR}/feature_tests"

EXIT_OK=0
EXIT_TEST_FAILED=1
EXIT_USAGE=2

LOG_DIR=""
TIMEOUT_SECONDS=0
RESULT_JSON=""
LIST_ONLY=0
REQUESTED_CASES=()
REGISTERED_CASES=()
SELECTED_CASES=()
RESULT_ENTRIES=()

usage() {
    cat <<'EOF'
用法: bash run_all_st.sh [日志目录] [选项]

选项:
  --log-dir 目录       日志保存目录（等价于位置参数）
  --case 用例名        只执行指定用例，可重复传入；名字带不带 .sh 后缀均可
  --timeout 秒数       单用例超时秒数，超时杀掉整个进程组；0=不限时（默认）
  --result-json 文件   逐用例结果写入该 JSON 文件，供 CI/脚本消费
  --list               只列出 test_cases.txt 中已启用的用例，不执行
  -h, --help           显示本帮助

日志目录：每个用例一个子目录（用例名去掉 .sh），内含 stdout.log / stderr.log
退出码: 0=全部通过  1=有用例失败或超时  2=用法或配置错误
EOF
}

# 日志目录只允许指定一次：重复指定时静默取其中一个会掩盖调用方的拼错
set_log_dir() {
    if [ -n "$LOG_DIR" ]; then
        echo "错误: 日志目录重复指定: $LOG_DIR 与 $1" >&2; exit "$EXIT_USAGE"
    fi
    LOG_DIR="$1"
}

parse_args() {
    while [ $# -gt 0 ]; do
        case "$1" in
            --log-dir)
                [ $# -ge 2 ] || { echo "错误: --log-dir 缺少参数" >&2; exit "$EXIT_USAGE"; }
                set_log_dir "$2"; shift 2 ;;
            --case)
                [ $# -ge 2 ] || { echo "错误: --case 缺少参数" >&2; exit "$EXIT_USAGE"; }
                REQUESTED_CASES+=("$2"); shift 2 ;;
            --timeout)
                [ $# -ge 2 ] || { echo "错误: --timeout 缺少参数" >&2; exit "$EXIT_USAGE"; }
                case "$2" in
                    ''|*[!0-9]*)
                        echo "错误: --timeout 需要非负整数秒数, 收到: $2" >&2
                        exit "$EXIT_USAGE" ;;
                esac
                TIMEOUT_SECONDS="$2"; shift 2 ;;
            --result-json)
                [ $# -ge 2 ] || { echo "错误: --result-json 缺少参数" >&2; exit "$EXIT_USAGE"; }
                RESULT_JSON="$2"; shift 2 ;;
            --list)
                LIST_ONLY=1; shift ;;
            -h|--help)
                usage; exit "$EXIT_OK" ;;
            -*)
                echo "错误: 未知参数: $1" >&2; usage >&2; exit "$EXIT_USAGE" ;;
            *)
                # 位置参数即日志目录，兼容旧用法
                set_log_dir "$1"; shift ;;
        esac
    done
}

# 读取 test_cases.txt 中启用的用例到 REGISTERED_CASES，保持注册顺序
load_cases() {
    REGISTERED_CASES=()
    local line
    while IFS= read -r line; do
        # 跳过注释和空行
        case "$line" in \#*|"") continue ;; esac
        REGISTERED_CASES+=("$line")
    done < "$CASE_FILE"
}

# 按 --case 过滤到 SELECTED_CASES。请求了未注册（或被注释禁用）的用例直接报错退出，
# 避免静默漏跑把「没跑」当「通过」；未指定 --case 时按注册顺序全选。
select_cases() {
    local -a enabled=("$@")
    if [ ${#REQUESTED_CASES[@]} -eq 0 ]; then
        SELECTED_CASES=("${enabled[@]}")
        return 0
    fi

    local -a missing=()
    local name req want hit
    for req in "${REQUESTED_CASES[@]}"; do
        want="${req%.sh}"
        hit=0
        for name in "${enabled[@]}"; do
            [ "${name%.sh}" = "$want" ] && { hit=1; break; }
        done
        [ "$hit" -eq 1 ] || missing+=("$req")
    done
    if [ ${#missing[@]} -gt 0 ]; then
        echo "错误: 以下用例未在 test_cases.txt 中注册或被注释禁用: ${missing[*]}" >&2
        echo "提示: 用 --list 查看当前启用的用例。" >&2
        exit "$EXIT_USAGE"
    fi

    SELECTED_CASES=()
    for name in "${enabled[@]}"; do
        for req in "${REQUESTED_CASES[@]}"; do
            [ "${name%.sh}" = "${req%.sh}" ] && { SELECTED_CASES+=("$name"); break; }
        done
    done
}

prepare_log_dir() {
    [ -z "$LOG_DIR" ] && return 0
    if ! mkdir -p "$LOG_DIR"; then
        echo "错误: 无法创建日志目录: $LOG_DIR" >&2
        exit "$EXIT_USAGE"
    fi
}

# 执行单个用例，结果写入全局 CASE_STATUS / CASE_EXIT_CODE / CASE_LOG_DIR。
# 日志目录下按用例建子目录，stdout/stderr 分别落盘并回显终端。
# 用例脚本不存在时记 skipped：各分支用例集本就不同，缺用例不该算失败。
run_case() {
    local name="$1"
    CASE_STATUS="failed"
    CASE_EXIT_CODE=""
    CASE_LOG_DIR=""

    if [ ! -f "${TEST_DIR}/${name}" ]; then
        CASE_STATUS="skipped"
        return 0
    fi

    local case_dir=""
    if [ -n "$LOG_DIR" ]; then
        # 用例名去掉 .sh 后缀作子目录名，如 case_a.sh -> <LOG_DIR>/case_a/
        case_dir="${LOG_DIR}/${name%.sh}"
        if ! mkdir -p "$case_dir"; then
            echo "错误: 无法创建用例日志目录: $case_dir" >&2
            exit "$EXIT_USAGE"
        fi
        CASE_LOG_DIR="$case_dir"
    fi

    # timeout 把用例放进独立进程组并在超时时杀掉整组；只杀 bash 会留下占着 NPU 的孤儿
    # torchrun 进程，污染后续用例。--kill-after 兜住忽略 TERM 的进程。
    local -a cmd=()
    [ "$TIMEOUT_SECONDS" -gt 0 ] && cmd=(timeout --signal=TERM --kill-after=30 "$TIMEOUT_SECONDS")
    cmd+=(bash -e "${TEST_DIR}/${name}")

    local rc=0
    # < /dev/null 防止子进程读走注册文件剩余内容
    if [ -n "$case_dir" ]; then
        # 进程替换: 两路分别 tee 落盘，同时保留终端回显
        MEGATRON_NIGHTLY_LOG_DIR="$case_dir" \
            "${cmd[@]}" < /dev/null \
            > >(tee "${case_dir}/stdout.log") \
            2> >(tee "${case_dir}/stderr.log" >&2)
        rc=$?
    else
        "${cmd[@]}" < /dev/null
        rc=$?
    fi

    CASE_EXIT_CODE="$rc"
    case "$rc" in
        0) CASE_STATUS="passed" ;;
        124|137) CASE_STATUS="timeout" ;;
        *) CASE_STATUS="failed" ;;
    esac
}

# 转义 JSON 字符串中的反斜杠、引号与控制字符（用例名、路径均来自外部输入）
json_escape() {
    local s="$1"
    s="${s//\\/\\\\}"
    s="${s//\"/\\\"}"
    s="${s//$'\n'/\\n}"
    s="${s//$'\r'/\\r}"
    s="${s//$'\t'/\\t}"
    printf '%s' "$s"
}

# 构造单个用例的 JSON 对象；未产生的字段写 null
json_case_entry() {
    local name="$1" status="$2" exit_code="$3" duration="$4" log_dir="$5"
    local code_field="null" log_field="null"
    [ -n "$exit_code" ] && code_field="$exit_code"
    [ -n "$log_dir" ] && log_field="\"$(json_escape "$log_dir")\""
    printf '{"test_id":"%s","name":"%s","status":"%s","exit_code":%s,"duration_seconds":%s,"log_dir":%s}' \
        "$(json_escape "${name%.sh}")" "$(json_escape "$name")" "$status" \
        "$code_field" "$duration" "$log_field"
}

# 写出逐用例结果；未指定 --result-json 时为空操作
write_result_json() {
    [ -z "$RESULT_JSON" ] && return 0
    local total="$1" passed="$2" failed="$3" timedout="$4" skipped="$5"
    local log_field="null" out_dir i last

    [ -n "$LOG_DIR" ] && log_field="\"$(json_escape "$LOG_DIR")\""
    out_dir="$(dirname "$RESULT_JSON")"
    if ! mkdir -p "$out_dir"; then
        echo "错误: 无法创建结果文件目录: $out_dir" >&2
        exit "$EXIT_USAGE"
    fi

    last=$((${#RESULT_ENTRIES[@]} - 1))
    {
        printf '{\n'
        printf '  "total": %s,\n' "$total"
        printf '  "passed": %s,\n' "$passed"
        printf '  "failed": %s,\n' "$failed"
        printf '  "timeout": %s,\n' "$timedout"
        printf '  "skipped": %s,\n' "$skipped"
        printf '  "timeout_seconds": %s,\n' "$TIMEOUT_SECONDS"
        printf '  "log_dir": %s,\n' "$log_field"
        printf '  "cases": [\n'
        for i in "${!RESULT_ENTRIES[@]}"; do
            if [ "$i" -lt "$last" ]; then
                printf '    %s,\n' "${RESULT_ENTRIES[$i]}"
            else
                printf '    %s\n' "${RESULT_ENTRIES[$i]}"
            fi
        done
        printf '  ]\n'
        printf '}\n'
    } > "$RESULT_JSON" || {
        echo "错误: 无法写入结果文件: $RESULT_JSON" >&2
        exit "$EXIT_USAGE"
    }
    echo "Result JSON: $RESULT_JSON"
}

main() {
    parse_args "$@"
    prepare_log_dir

    if [ ! -f "$CASE_FILE" ]; then
        echo "错误: 用例注册文件不存在: $CASE_FILE" >&2
        exit "$EXIT_USAGE"
    fi

    load_cases

    if [ "$LIST_ONLY" -eq 1 ]; then
        [ ${#REGISTERED_CASES[@]} -gt 0 ] && printf '%s\n' "${REGISTERED_CASES[@]}"
        return "$EXIT_OK"
    fi

    select_cases "${REGISTERED_CASES[@]}"

    if [ ${#SELECTED_CASES[@]} -eq 0 ]; then
        echo "INFO: 没有需要执行的用例。"
        return "$EXIT_OK"
    fi

    local passed=0 failed=0 timedout=0 skipped=0 total=0 results=""
    local name elapsed start start_all duration
    start_all=$(date +%s)

    for name in "${SELECTED_CASES[@]}"; do
        total=$((total + 1))
        start=$(date +%s)
        echo "[$(date +%H:%M:%S)] start: $name"

        run_case "$name"
        duration=$(( $(date +%s) - start ))

        case "$CASE_STATUS" in
            passed)  passed=$((passed + 1)) ;;
            timeout) timedout=$((timedout + 1)) ;;
            skipped) skipped=$((skipped + 1)) ;;
            *)       failed=$((failed + 1)) ;;
        esac

        if [ "$CASE_STATUS" = "skipped" ]; then
            echo "[$(date +%H:%M:%S)] skipped: $name (脚本不存在)"
        else
            echo "[$(date +%H:%M:%S)] $CASE_STATUS: $name (${duration}s)"
        fi
        results="${results}${CASE_STATUS}  ${name}  ${duration}s\n"
        RESULT_ENTRIES+=("$(json_case_entry "$name" "$CASE_STATUS" "$CASE_EXIT_CODE" "$duration" "$CASE_LOG_DIR")")
        echo ""
    done

    echo "========================================"
    echo "Total: $total  passed: $passed  failed: $failed  timeout: $timedout  skipped: $skipped"
    echo "Total time: $(( $(date +%s) - start_all ))s"
    if [ -n "$LOG_DIR" ]; then
        echo "Log dir: $LOG_DIR"
    fi
    echo "========================================"
    echo -e "$results"

    write_result_json "$total" "$passed" "$failed" "$timedout" "$skipped"

    [ $((failed + timedout)) -gt 0 ] && exit "$EXIT_TEST_FAILED"
    exit "$EXIT_OK"
}

main "$@"
