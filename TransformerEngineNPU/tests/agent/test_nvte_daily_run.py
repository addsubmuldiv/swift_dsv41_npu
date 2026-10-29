import json
import sys
from datetime import datetime, timezone
from pathlib import Path

SCRIPT_DIR = (
    Path(__file__).resolve().parents[2]
    / ".agent"
    / "skills"
    / "tenpu-daily-orchestrator"
    / "scripts"
)
sys.path.insert(0, str(SCRIPT_DIR))

from nvte_daily_run import (  # noqa: E402
    commit_and_push,
    compute_search_window,
    compute_target_date,
    fetch_merged_prs,
    load_token,
    main,
    save_json,
)


def test_target_date_previous_utc_day():
    ref = datetime(2026, 8, 6, 0, 30, tzinfo=timezone.utc)
    assert compute_target_date(ref) == "2026-08-05"


def test_target_date_just_after_utc_midnight():
    ref = datetime(2026, 8, 6, 0, 0, 1, tzinfo=timezone.utc)
    assert compute_target_date(ref) == "2026-08-05"


def test_target_date_month_crossing():
    ref = datetime(2026, 8, 1, 12, 0, tzinfo=timezone.utc)
    assert compute_target_date(ref) == "2026-07-31"


def test_target_date_year_crossing():
    ref = datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)
    assert compute_target_date(ref) == "2025-12-31"


def test_target_date_beijing_not_utc():
    """区分时区的边界：UTC 8/5 23:00 已是北京 8/6，前一天应为 8/5（UTC 语义会是 8/4）。"""
    ref = datetime(2026, 8, 5, 23, 0, tzinfo=timezone.utc)
    assert compute_target_date(ref) == "2026-08-05"


def test_search_window():
    assert compute_search_window("2026-08-05") == (
        "2026-08-04T16:00:00Z",
        "2026-08-05T16:00:00Z",
    )


def test_search_window_month_end():
    assert compute_search_window("2026-07-31") == (
        "2026-07-30T16:00:00Z",
        "2026-07-31T16:00:00Z",
    )


def test_save_json_roundtrip(tmp_path):
    path = tmp_path / "a" / "b.json"
    save_json(path, {"n": 1})
    assert json.loads(path.read_text(encoding="utf-8")) == {"n": 1}


def test_load_token_from_environment(monkeypatch):
    monkeypatch.setenv("GH_TOKEN", "test-token")
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)

    assert load_token() == "test-token"


def test_fetch_merged_prs_commit_based(monkeypatch):
    """按 main 分支 commit 历史过滤北京自然日窗口，从 message 提取 PR 号并补 files。"""
    commits = [
        # 窗口内（newest first）：#3153 08-05 15:27 UTC
        {
            "sha": "bf4b2b9",
            "commit": {
                "message": "Add TensorProto mechanism (#3153)\n\nbody",
                "committer": {"date": "2026-08-05T15:27:31Z"},
            },
        },
        # 窗口内边界：#3314 08-04 16:15 UTC（北京 8/5 00:15，应在窗口内）
        {
            "sha": "82a0f6e",
            "commit": {
                "message": "NLTK pin (#3314)",
                "committer": {"date": "2026-08-04T16:15:33Z"},
            },
        },
        # 窗口外：#3000 08-04 15:59 UTC（北京 8/4 23:59）—— 必须被过滤
        {
            "sha": "old",
            "commit": {"message": "older (#3000)", "committer": {"date": "2026-08-04T15:59:59Z"}},
        },
    ]

    def fake_fetch_json(url, token, params=None):
        if url.endswith("/commits"):
            return commits  # 一页返回全部；末条 committer < start 触发停止
        if "/files" in url:
            return [{"filename": "a.py", "status": "modified"}]
        if url.endswith("/pulls/3153"):
            return {
                "number": 3153,
                "title": "t1",
                "merged_at": "2026-08-05T15:27:31Z",
                "html_url": "u1",
            }
        if url.endswith("/pulls/3314"):
            return {
                "number": 3314,
                "title": "t2",
                "merged_at": "2026-08-04T16:15:33Z",
                "html_url": "u2",
            }
        raise AssertionError(f"unexpected url: {url}")

    monkeypatch.setattr("nvte_daily_run.fetch_json", fake_fetch_json)
    prs = fetch_merged_prs("t", "2026-08-05")
    assert [p["number"] for p in prs] == [3153, 3314]
    assert prs[0]["merged_at"] == "2026-08-05T15:27:31Z"
    assert prs[1]["files"] == [{"filename": "a.py", "status": "modified"}]


def test_main_saves_input_locally(tmp_path, monkeypatch):
    """抓取结果只写入 TENPU 本地 output 目录。"""

    def fake_fetch(token, date_str):
        return [{"number": 1, "title": "t"}]

    monkeypatch.setattr("nvte_daily_run.load_token", lambda: "t")
    monkeypatch.setattr("nvte_daily_run.fetch_merged_prs", fake_fetch)
    monkeypatch.setattr("nvte_daily_run.OUTPUT_DIR", tmp_path)

    rc = main(["--date", "2026-08-05"])
    assert rc == 0
    data = json.loads((tmp_path / "2026-08-05" / "prs.json").read_text(encoding="utf-8"))
    assert data[0]["number"] == 1


def test_main_fetch_idempotent_skips_fetch(tmp_path, monkeypatch):
    """prs.json 已存在 → 跳过抓取（幂等，支持当日重跑）。"""
    out = tmp_path / "2026-08-05"
    out.mkdir(parents=True)
    (out / "prs.json").write_text("[]", encoding="utf-8")

    def fake_fetch(token, date_str):
        raise AssertionError("不应重复抓取")

    monkeypatch.setattr("nvte_daily_run.fetch_merged_prs", fake_fetch)
    monkeypatch.setattr("nvte_daily_run.OUTPUT_DIR", tmp_path)

    rc = main(["--date", "2026-08-05"])
    assert rc == 0


def test_main_push_without_report_fails(tmp_path, monkeypatch):
    monkeypatch.setattr("nvte_daily_run.load_token", lambda: "t")
    monkeypatch.setattr("nvte_daily_run.fetch_merged_prs", lambda token, date: [])
    monkeypatch.setattr("nvte_daily_run.OUTPUT_DIR", tmp_path)

    assert main(["--date", "2026-08-05", "--push"]) == 1


def test_main_push_dry_run_does_not_commit(tmp_path, monkeypatch):
    out = tmp_path / "2026-08-05"
    out.mkdir(parents=True)
    (out / "prs.json").write_text("[]", encoding="utf-8")
    (out / "report.md").write_text("# report", encoding="utf-8")
    calls = []

    monkeypatch.setattr("nvte_daily_run.OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(
        "nvte_daily_run.commit_and_push",
        lambda date, report, data: calls.append(date),
    )

    assert main(["--date", "2026-08-05", "--push", "--dry-run"]) == 0
    assert not calls


def test_commit_and_push_is_idempotent(tmp_path, monkeypatch):
    worktree = tmp_path / "datasets"
    (worktree / ".git").mkdir(parents=True)
    report = tmp_path / "report.md"
    data = tmp_path / "prs.json"
    report.write_text("# report", encoding="utf-8")
    data.write_text("[]", encoding="utf-8")
    calls = []

    monkeypatch.setattr("nvte_daily_run.DATASETS_WORKTREE", worktree)
    monkeypatch.setattr("nvte_daily_run.run_git", lambda cwd, args: calls.append(args))
    monkeypatch.setattr("nvte_daily_run.has_staged_changes", lambda cwd: False)

    commit_and_push("2026-08-05", report, data)

    assert calls == [["add", "nvte-daily/"]]
