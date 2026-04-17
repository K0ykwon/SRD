"""Tests for the required reproduction audit helpers."""

from pathlib import Path

from srd.eval.reproduction_audit import audit_score_and_aggregate


def test_reproduction_audit_writes_expected_files_and_summary(tmp_path: Path) -> None:
    paths = audit_score_and_aggregate(tmp_path)

    assert Path(paths["aggregate_csv"]).exists()
    assert Path(paths["grouped_csv"]).exists()
    summary_path = Path(paths["summary_markdown"])
    assert summary_path.exists()

    summary = summary_path.read_text(encoding="utf-8")
    assert "Score And Aggregate Audit" in summary
    assert "delayed_kv" in summary
    assert "needle" in summary
    assert "delayed_copy" in summary
    assert "observed `0.5000`, expected `0.5000`" in summary
