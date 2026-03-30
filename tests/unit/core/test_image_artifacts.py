"""Unit tests for core/image_artifacts.py."""
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

from core.image_artifacts import extract_image_artifacts_from_tool_records, resolve_local_image_paths


def test_extract_image_artifacts_from_nested_payload():
    records = [
        {
            "tool_name": "web_search",
            "result_summary": {
                "results": [
                    {"title": "a", "image_url": "https://images.unsplash.com/photo-1.png"},
                    {"title": "b", "thumbnail": "https://cdn.search.brave.com/img.webp"},
                ]
            },
        }
    ]

    artifacts = extract_image_artifacts_from_tool_records(records)

    assert len(artifacts) == 2
    assert artifacts[0]["type"] == "image"
    assert artifacts[0]["source"] == "searched"
    assert artifacts[0]["trust"] == "untrusted"


def test_extract_image_artifacts_marks_image_gen_as_generated():
    records = [
        {
            "tool_name": "image_gen",
            "result_summary": "Saved to assets/avatar_fullbody.png",
        }
    ]

    artifacts = extract_image_artifacts_from_tool_records(records)

    assert len(artifacts) == 1
    assert artifacts[0]["source"] == "generated"
    assert artifacts[0]["trust"] == "trusted"
    assert artifacts[0]["path"] == "assets/avatar_fullbody.png"


def test_extract_image_artifacts_ignores_non_skill_urls():
    records = [
        {
            "tool_name": "chatwork_rooms",
            "result_summary": {
                "rooms": [
                    {"icon_path": "https://appdata.chatwork.com/icon/ico_group.png"},
                    {"name": "x", "url": "https://appdata.chatwork.com/avatar/u.jpg"},
                    {"name": "y", "url": "https://images.unsplash.com/photo-1.png"},
                ]
            },
        }
    ]

    artifacts = extract_image_artifacts_from_tool_records(records)

    assert len(artifacts) == 1
    assert artifacts[0]["source"] == "searched"
    assert artifacts[0]["url"] == "https://images.unsplash.com/photo-1.png"


# ── resolve_local_image_paths tests ──────────────────────


def test_resolve_local_image_paths_file_url(tmp_path: Path):
    """file:// URLs are copied to attachments/ and markdown is rewritten."""
    img = tmp_path / "photo.jpg"
    img.write_bytes(b"\xff\xd8\xff dummy jpeg")

    anima_dir = tmp_path / "anima"
    anima_dir.mkdir()

    text = f"Look at this: ![pic](file://{img})"
    rewritten, artifacts = resolve_local_image_paths(text, anima_dir)

    assert "attachments/photo.jpg" in rewritten
    assert "file://" not in rewritten
    assert len(artifacts) == 1
    assert artifacts[0]["path"] == "attachments/photo.jpg"
    assert (anima_dir / "attachments" / "photo.jpg").exists()


def test_resolve_local_image_paths_absolute_path(tmp_path: Path):
    """Bare absolute paths are resolved the same way."""
    img = tmp_path / "image.png"
    img.write_bytes(b"\x89PNG\r\n\x1a\n dummy png")

    anima_dir = tmp_path / "anima"
    anima_dir.mkdir()

    text = f"![alt text]({img})"
    rewritten, artifacts = resolve_local_image_paths(text, anima_dir)

    assert "attachments/image.png" in rewritten
    assert str(img) not in rewritten
    assert len(artifacts) == 1
    assert (anima_dir / "attachments" / "image.png").exists()


def test_resolve_local_image_paths_missing_file(tmp_path: Path):
    """Non-existent files are left as-is."""
    anima_dir = tmp_path / "anima"
    anima_dir.mkdir()

    text = "![gone](/nonexistent/path/image.jpg)"
    rewritten, artifacts = resolve_local_image_paths(text, anima_dir)

    assert rewritten == text
    assert artifacts == []


def test_resolve_local_image_paths_no_images():
    """Text without markdown images passes through unchanged."""
    text = "Hello, no images here."
    rewritten, artifacts = resolve_local_image_paths(text, Path("/tmp/fake"))

    assert rewritten == text
    assert artifacts == []


def test_resolve_local_image_paths_non_image_extension(tmp_path: Path):
    """Non-image files are not processed."""
    txt = tmp_path / "doc.txt"
    txt.write_text("hello")

    anima_dir = tmp_path / "anima"
    anima_dir.mkdir()

    text = f"![doc]({txt})"
    rewritten, artifacts = resolve_local_image_paths(text, anima_dir)

    assert rewritten == text
    assert artifacts == []


def test_resolve_local_image_paths_dedup(tmp_path: Path):
    """Same file referenced twice produces only one copy."""
    img = tmp_path / "dup.jpg"
    img.write_bytes(b"\xff\xd8\xff dup")

    anima_dir = tmp_path / "anima"
    anima_dir.mkdir()

    text = f"![a]({img}) and ![b]({img})"
    rewritten, artifacts = resolve_local_image_paths(text, anima_dir)

    assert rewritten.count("attachments/dup.jpg") == 2
    assert len(artifacts) == 1


def test_resolve_local_image_paths_mixed(tmp_path: Path):
    """Mix of local, relative, and http images — only local ones are resolved."""
    img = tmp_path / "local.png"
    img.write_bytes(b"\x89PNG\r\n\x1a\n data")

    anima_dir = tmp_path / "anima"
    anima_dir.mkdir()

    text = f"![local]({img}) ![relative](attachments/foo.png) ![remote](https://example.com/img.png)"
    rewritten, artifacts = resolve_local_image_paths(text, anima_dir)

    assert "attachments/local.png" in rewritten
    assert "attachments/foo.png" in rewritten
    assert "https://example.com/img.png" in rewritten
    assert len(artifacts) == 1


def test_resolve_local_image_paths_symlink_rejected(tmp_path: Path):
    """Symlinks are rejected to prevent unintended file reads."""
    real_img = tmp_path / "secret.png"
    real_img.write_bytes(b"\x89PNG\r\n\x1a\n secret")
    link = tmp_path / "link.png"
    link.symlink_to(real_img)

    anima_dir = tmp_path / "anima"
    anima_dir.mkdir()

    text = f"![link]({link})"
    rewritten, artifacts = resolve_local_image_paths(text, anima_dir)

    assert rewritten == text
    assert artifacts == []


def test_resolve_local_image_paths_respects_max_limit(tmp_path: Path):
    """At most _MAX_ARTIFACTS_PER_RESPONSE images are resolved."""
    anima_dir = tmp_path / "anima"
    anima_dir.mkdir()

    parts = []
    for i in range(8):
        img = tmp_path / f"img{i}.jpg"
        img.write_bytes(b"\xff\xd8\xff data")
        parts.append(f"![{i}]({img})")
    text = " ".join(parts)

    rewritten, artifacts = resolve_local_image_paths(text, anima_dir)

    assert len(artifacts) == 5
    assert rewritten.count("attachments/") == 5
