import json

import pytest

from playwright_mcp.server import (
    _load_storage_state_file,
    _prepare_storage_origins,
    _resolve_storage_state_path,
)


def test_resolve_storage_state_path_allows_relative_path_under_cwd(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    storage_path = tmp_path / "output" / "playwright" / "auth" / "linkedin_state.json"
    storage_path.parent.mkdir(parents=True)
    storage_path.write_text('{"cookies": [], "origins": []}', encoding="utf-8")

    assert (
        _resolve_storage_state_path("output/playwright/auth/linkedin_state.json")
        == storage_path.resolve()
    )


def test_resolve_storage_state_path_rejects_outside_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    outside_path = tmp_path.parent / "outside_state.json"
    outside_path.write_text('{"cookies": [], "origins": []}', encoding="utf-8")

    try:
        with pytest.raises(PermissionError):
            _resolve_storage_state_path(str(outside_path))
    finally:
        outside_path.unlink(missing_ok=True)


def test_load_storage_state_file_normalizes_missing_collections(tmp_path):
    storage_path = tmp_path / "state.json"
    storage_path.write_text("{}", encoding="utf-8")

    storage_state = _load_storage_state_file(storage_path)

    assert storage_state["cookies"] == []
    assert storage_state["origins"] == []


def test_prepare_storage_origins_filters_to_requested_origins():
    origins = [
        {
            "origin": "https://www.linkedin.com",
            "localStorage": [{"name": "token", "value": "secret"}],
        },
        {
            "origin": "https://example.com",
            "localStorage": [{"name": "other", "value": "value"}],
        },
    ]

    prepared = _prepare_storage_origins(
        origins,
        ["https://www.linkedin.com/feed/"],
    )

    assert prepared == [
        ("https://www.linkedin.com", [{"name": "token", "value": "secret"}])
    ]


def test_load_storage_state_file_rejects_invalid_top_level_json(tmp_path):
    storage_path = tmp_path / "state.json"
    storage_path.write_text(json.dumps([]), encoding="utf-8")

    with pytest.raises(ValueError, match="JSON object"):
        _load_storage_state_file(storage_path)
