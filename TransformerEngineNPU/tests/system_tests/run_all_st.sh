#!/bin/bash
# run_all_st.sh - 一键执行所有ST用例
#
# 用法:
#   bash run_all_st.sh [日志目录]
#   第一个参数为用例日志保存目录，日志文件名为 <用例名(去掉.sh)>.log；不传则不记录日志
#
# 用例在 test_cases.txt 中注册，每行一个，按文件顺序执行

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CASE_FILE="${SCRIPT_DIR}/test_cases.txt"
TEST_DIR="${SCRIPT_DIR}/feature_tests"
LOG_DIR=""

parse_args() {
    if [ $# -gt 1 ]; then
        echo "用法: bash run_all_st.sh [日志目录]" >&2
        exit 1
    fi
    LOG_DIR="${1:-}"
}

prepare_log_dir() {
    [ -z "$LOG_DIR" ] && return 0
    if ! mkdir -p "$LOG_DIR"; then
        echo "错误: 无法创建日志目录: $LOG_DIR" >&2
        exit 1
    fi
}

# 执行单个用例；指定日志目录时用 tee 双写（stdout+stderr），
# < /dev/null 防止子进程读走注册文件剩余内容
run_case() {
    local name="$1"
    if [ -n "$LOG_DIR" ]; then
        # 日志名去掉用例的 .sh 后缀，如 case_a.sh -> case_a.log
        bash -e "${TEST_DIR}/${name}" < /dev/null 2>&1 | tee "${LOG_DIR}/${name%.sh}.log"
        return "${PIPESTATUS[0]}"
    fi
    bash -e "${TEST_DIR}/${name}" < /dev/null
}

main() {
    parse_args "$@"
    prepare_log_dir

    local passed=0 failed=0 total=0 results=""
    local name result elapsed start start_all
    start_all=$(date +%s)

    while read -r name; do
        # 跳过注释和空行
        case "$name" in \#*|"") continue ;; esac

        total=$((total + 1))
        start=$(date +%s)
        echo "[$(date +%H:%M:%S)] start: $name"

        if run_case "$name"; then
            result=passed
            passed=$((passed + 1))
        else
            result=failed
            failed=$((failed + 1))
        fi
        elapsed=$(( $(date +%s) - start ))
        echo "[$(date +%H:%M:%S)] $result: $name (${elapsed}s)"
        results="${results}${result}  ${name}  ${elapsed}s\n"
        echo ""
    done < "$CASE_FILE"

    echo "========================================"
    echo "Total: $total  passed: $passed  failed: $failed"
    echo "Total time: $(( $(date +%s) - start_all ))s"
    if [ -n "$LOG_DIR" ]; then
        echo "Log dir: $LOG_DIR"
    fi
    echo "========================================"
    echo -e "$results"

    [ $failed -gt 0 ] && exit 1
    exit 0
}

main "$@"
