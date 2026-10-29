import sys
from pathlib import Path

SCRIPT_DIR = (
    Path(__file__).resolve().parents[2]
    / ".agent"
    / "skills"
    / "tenpu-daily-orchestrator"
    / "scripts"
)
sys.path.insert(0, str(SCRIPT_DIR))

import nvte_daily_orchestrate as orchestrate  # noqa: E402


def write_report(root: Path, date_str: str, final_status: str) -> Path:
    report = root / date_str / "report.md"
    report.parent.mkdir(parents=True)
    report.write_text(
        f"# report\n\n```yaml\ndaily_te_npu:\n  final_status: {final_status}\n```\n",
        encoding="utf-8",
    )
    return report


def test_extract_final_status_accepts_local_report_states(tmp_path, monkeypatch):
    monkeypatch.setattr(orchestrate, "OUTPUT_DIR", tmp_path)

    for status in ("analysis-complete", "no-op", "blocked", "needs-review"):
        write_report(tmp_path, status, status)
        assert orchestrate.extract_final_status(status) == status


def test_existing_report_is_not_overwritten_and_is_synced(tmp_path, monkeypatch):
    date_str = "2026-08-05"
    report = write_report(tmp_path, date_str, "analysis-complete")
    original = report.read_text(encoding="utf-8")

    monkeypatch.setattr(orchestrate, "OUTPUT_DIR", tmp_path)
    calls = []
    monkeypatch.setattr(
        orchestrate,
        "run_runner",
        lambda date, push=False: calls.append((date, push)) or 0,
    )
    monkeypatch.setattr(orchestrate, "log", lambda message, date: None)

    assert orchestrate.main(["--date", date_str]) == 0
    assert report.read_text(encoding="utf-8") == original
    assert calls == [(date_str, False), (date_str, True)]


def test_skip_agent_validates_existing_local_report(tmp_path, monkeypatch):
    date_str = "2026-08-05"
    write_report(tmp_path, date_str, "needs-review")

    monkeypatch.setattr(orchestrate, "OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(orchestrate, "run_runner", lambda date, push=False: 0)
    monkeypatch.setattr(orchestrate, "log", lambda message, date: None)

    assert orchestrate.main(["--date", date_str, "--force", "--skip-agent"]) == 0


def test_invalid_local_report_fails_validation(tmp_path, monkeypatch):
    date_str = "2026-08-05"
    report = tmp_path / date_str / "report.md"
    report.parent.mkdir(parents=True)
    report.write_text("# missing machine block\n", encoding="utf-8")

    monkeypatch.setattr(orchestrate, "OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(orchestrate, "run_runner", lambda date, push=False: 0)
    monkeypatch.setattr(orchestrate, "log", lambda message, date: None)

    assert orchestrate.main(["--date", date_str, "--force", "--skip-agent"]) == 1
