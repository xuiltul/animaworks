from __future__ import annotations

from core.response_normalize import normalize_user_facing_response_text


def test_extracts_post_channel_text() -> None:
    raw = '{"name":"post_channel","arguments":{"channel":"ops","text":"@cmnt\\n\\n確認しました。"}}'
    assert normalize_user_facing_response_text(raw) == "@cmnt\n\n確認しました。"


def test_extracts_nested_function_text() -> None:
    raw = '{"id":"call_1","type":"function","function":{"name":"send_message","arguments":{"to":"cmnt","content":"了解しました。"}}}'
    assert normalize_user_facing_response_text(raw) == "了解しました。"


def test_extracts_prefixed_tool_call_text() -> None:
    raw = '了解しました。以下の方法で回答します。\n\n{"name":"send_message","arguments":{"to":"cmnt","content":"本文です。"}}'
    assert normalize_user_facing_response_text(raw) == "本文です。"


def test_summarizes_status_payload_in_japanese() -> None:
    raw = (
        '{"status":"idle","file_access_verified":true,'
        '"paths_verified":["E:\\\\OneDriveBiz\\\\Tools\\\\General"],'
        '"access_method":"read_file",'
        '"last_verified_file":"E:\\\\OneDriveBiz\\\\Tools\\\\General\\\\sample.txt",'
        '"note":"Windows 環境でも直接読み取り可能。"}'
    )
    normalized = normalize_user_facing_response_text(raw)
    assert "状態: 待機中" in normalized
    assert "ファイルアクセス確認: 可能" in normalized
    assert "確認方法: read_file" in normalized
    assert "最終確認ファイル: E:\\OneDriveBiz\\Tools\\General\\sample.txt" in normalized
    assert "- E:\\OneDriveBiz\\Tools\\General" in normalized
    assert "補足: Windows 環境でも直接読み取り可能。" in normalized


def test_preserves_error_message_details() -> None:
    raw = (
        '{"status":"error","error_type":"FileNotFound",'
        '"message":"File not found: E:\\\\OneDriveBiz\\\\Tools\\\\General\\\\openclaw-ops-dashboard\\\\dashboard.py",'
        '"suggestion":"Use list_directory to find the correct path"}'
    )
    normalized = normalize_user_facing_response_text(raw)
    assert "エラー種別: FileNotFound" in normalized
    assert "File not found: E:\\OneDriveBiz\\Tools\\General\\openclaw-ops-dashboard\\dashboard.py" in normalized
    assert "対処候補: Use list_directory to find the correct path" in normalized


def test_leaves_unrecognized_json_unchanged() -> None:
    raw = '{"foo":"bar"}'
    assert normalize_user_facing_response_text(raw) == raw
