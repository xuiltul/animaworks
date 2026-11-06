"""Structural host protocols for the compositional mixins."""

from __future__ import annotations

from typing import Any, Protocol

from core.tooling.tool_context import ToolContext


class _CommsToolsHost(Protocol):
    """Structural host members used by CommsToolsMixin."""

    _activity: Any
    _anima_dir: Any
    _anima_name: Any
    _build_send_feedback: Any
    _build_recipient_busy_notice: Any
    _call_human_keys: Any
    _channel_company_boundary_error: Any
    _cross_company_communication_error: Any
    _fanout_board_mentions: Any
    _fire_board_slack_sync: Any
    _human_notifier: Any
    _last_call_human_callback_id: Any
    _last_call_human_denied: Any
    _last_post_channel_succeeded: Any
    _messenger: Any
    _on_message_sent: Any
    _pending_notifications: Any
    _persist_replied_to: Any
    _posted_channels: Any
    _replied_to: Any
    _resolve_meeting_participant: Any
    _session_origin: Any
    _session_origin_chain: Any
    posted_channels_for: Any
    replied_to_for: Any
    session_id: Any


class _ChatFilesHost(Protocol):
    """Structural host members used by ChatFilesToolsMixin."""

    _anima_dir: Any
    _check_file_permission: Any
    _runtime_session_context: Any


class _CreateAnimaHost(Protocol):
    """Structural host members used by CreateAnimaMixin."""

    _tool_context: ToolContext


class _DelegationHost(Protocol):
    """Structural host members used by DelegationMixin."""

    _activity: Any
    _anima_dir: Any
    _anima_name: Any
    _check_subordinate: Any
    _messenger: Any
    _session_origin: Any
    _session_origin_chain: Any


class _ExecutionToolsHost(Protocol):
    """Structural host members used by ExecutionToolsMixin."""

    _tool_context: ToolContext


class _FileToolsHost(Protocol):
    """Structural host members used by FileToolsMixin."""

    _WEB_FETCH_CACHE_MAX_SIZE: Any
    _WEB_FETCH_CACHE_TTL: Any
    _WEB_FETCH_MAX_CHARS: Any
    _WEB_FETCH_MAX_REDIRECTS: Any
    _WEB_FETCH_SAFETY_NOTICE: Any
    _WEB_FETCH_TIMEOUT: Any
    _WEB_FETCH_USER_AGENT: Any
    _anima_dir: Any
    _check_file_permission: Any
    _context_window: Any
    _handle_list_directory: Any
    _is_private_host: Any
    _is_state_file: Any
    _iter_permitted_tree_paths: Any
    _load_permissions_config: Any
    _matches_recursive_pattern: Any
    _read_file_budget: Any
    _read_paths: Any
    _resolved_file_deny_roots: Any
    _state_file_lock: Any
    _try_write_with_frontmatter: Any
    _web_fetch_cache: Any
    _web_fetch_cache_lock: Any


class _MemoryToolsHost(Protocol):
    """Structural host members used by MemoryToolsMixin."""

    _USED_COLLECTION_PREFIXES: Any
    _activity: Any
    _anima_dir: Any
    _anima_name: Any
    _anima_search_hint: Any
    _check_file_permission: Any
    _check_tool_creation_permission: Any
    _collection_for_memory_file: Any
    _complete_memory_write: Any
    _current_anima_name: Any
    _descendant_activity_dirs: Any
    _descendant_state_dirs: Any
    _descendant_state_files: Any
    _format_memory_write_result: Any
    _format_search_results: Any
    _handle_post_channel: Any
    _handle_read_channel: Any
    _is_flat_personal_skill_path: Any
    _is_skill_path: Any
    _is_state_file: Any
    _load_permissions_config: Any
    _memory: Any
    _memory_source_is_denied: Any
    _memory_write_similarity_hint: Any
    _on_schedule_changed: Any
    _read_paths: Any
    _record_memory_file_change: Any
    _record_memory_file_used: Any
    _record_skill_view_if_applicable: Any
    _reload_schedule_if_needed: Any
    _resolve_external: Any
    _resolve_write_origin: Any
    _resolved_file_deny_roots: Any
    _state_file_lock: Any
    _subordinate_activity_dirs: Any
    _subordinate_management_files: Any
    _superuser: Any
    _update_longterm_bm25_source: Any
    _update_memory_write_indexes: Any
    _write_memory_scope: Any
    _write_plain_memory_file: Any


class _DashboardHost(Protocol):
    """Structural host members used by DashboardMixin."""

    _activity: Any
    _anima_name: Any
    _check_descendant: Any
    _get_all_descendants: Any
    _get_direct_subordinates: Any
    _parse_since: Any
    _process_supervisor: Any
    _read_recent_activity: Any
    _render_tree: Any


class _PermissionsHost(Protocol):
    """Structural host members used by PermissionsMixin."""

    _anima_dir: Any
    _anima_name: Any
    _descendant_activity_dirs: Any
    _descendant_state_dirs: Any
    _descendant_state_files: Any
    _dispatch: Any
    _external: Any
    _file_access_error: Any
    _load_permissions_config: Any
    _memory: Any
    _peer_activity_dirs: Any
    _resolved_file_deny_roots: Any
    _subordinate_activity_dirs: Any
    _subordinate_management_files: Any
    _subordinate_root_dirs: Any
    _superuser: Any
    _task_cwd: Any


class _SkillsToolsHost(Protocol):
    """Structural host members used by SkillsToolsMixin."""

    _activity: Any
    _anima_dir: Any
    _anima_name: Any
    _background_manager: Any
    _check_tool_creation_permission: Any
    _curator: Any
    _curator_index_entries: Any
    _external: Any
    _handle_curator_state_change: Any
    _mark_curator_reviewed: Any
    _memory: Any
    _read_paths: Any
    _scan_created_skill: Any
    _session_id: Any
    _session_origin: Any
    _trigger: Any


class _SubordinateControlHost(Protocol):
    """Structural host members used by SubordinateControlMixin."""

    _activity: Any
    _anima_name: Any
    _check_descendant: Any
    _get_all_descendants: Any
    _process_supervisor: Any
    _read_recent_activity: Any


class _WorkspaceToolsHost(Protocol):
    """Structural host members used by WorkspaceToolsMixin."""

    _anima_name: Any
    _session_origin: Any
    _session_origin_chain: Any
    _trigger: Any
    _workspace_grant_bool: Any
    _workspace_grant_has_human_origin: Any
    _workspace_grant_is_descendant: Any
    _workspace_grant_path_error: Any
    _workspace_grant_update_permissions: Any
    _workspace_grant_update_status: Any


class _OrgHelpersHost(Protocol):
    """Structural host members used by OrgHelpersMixin."""

    _anima_name: Any
    _get_all_descendants: Any
