"""Unit tests for the `match` last-run state helpers in cli.py.

File I/O is confined to pytest's tmp_path — no network, no Docker.
"""

from datetime import datetime, timezone

from app.cli import _read_last_match_run, _write_last_match_run

_TS = datetime(2026, 10, 5, 12, 0, 0, 123456, tzinfo=timezone.utc)


def test_state_roundtrips_with_microsecond_precision(tmp_path):
    path = tmp_path / "match_last_run.txt"

    _write_last_match_run(_TS, path)

    assert _read_last_match_run(path) == _TS


def test_write_creates_missing_parent_directory(tmp_path):
    path = tmp_path / "data" / "match_last_run.txt"

    _write_last_match_run(_TS, path)

    assert path.exists()


def test_read_returns_none_when_state_file_is_absent(tmp_path):
    assert _read_last_match_run(tmp_path / "missing.txt") is None


def test_read_returns_none_and_warns_on_garbage(tmp_path, capsys):
    path = tmp_path / "match_last_run.txt"
    path.write_text("not-a-timestamp", encoding="utf-8")

    assert _read_last_match_run(path) is None
    assert "Warning" in capsys.readouterr().err


def test_read_returns_none_and_warns_on_naive_timestamp(tmp_path, capsys):
    path = tmp_path / "match_last_run.txt"
    path.write_text("2026-10-05T12:00:00", encoding="utf-8")

    assert _read_last_match_run(path) is None
    assert "Warning" in capsys.readouterr().err
