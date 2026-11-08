# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
#
# This file is part of AnimaWorks core/server, licensed under Apache-2.0.
# See LICENSE for the full license text.

"""Central configuration module — facade re-exporting split modules."""

from __future__ import annotations

from core.config.anima_registry import (
    _resolve_supervisor_name,
    read_anima_company,
    read_anima_company_checked,
    read_anima_supervisor,
    register_anima_in_config,
    rename_anima_in_config,
    unregister_anima_from_config,
)
from core.config.helper_models import ResolvedHelperModel, resolve_helper_model, validate_helper_model_credentials
from core.config.io import get_config_path, invalidate_cache, load_config, save_config, update_config
from core.config.model_config import (
    DEFAULT_MAX_TOKENS,
    load_model_config,
    resolve_max_tokens,
    resolve_penalties,
    update_status_model,
)


def resolve_context_window(model_name: str, *args: object, **kwargs: object) -> int:
    """Re-export from ``core.prompt.context`` (lazy to avoid circular import)."""
    from core.prompt.context import resolve_context_window as _rcw

    return _rcw(model_name, *args, **kwargs)  # type: ignore[arg-type]


from core.config.model_mode import (
    KNOWN_MODELS,
    _load_models_json,
    _match_models_json,
    _match_pattern_table,
    _normalise_mode,
    _pattern_specificity,
    invalidate_models_json_cache,
    parse_fallback_entry,
    resolve_execution_mode,
)
from core.config.resolver import _load_status_json, resolve_anima_config
from core.config.schemas import (
    DEFAULT_ANIMA_MODEL,
    DEFAULT_LOCAL_LLM_BASE_URL,
    DEFAULT_LOCAL_LLM_MODEL,
    DEFAULT_LOCAL_LLM_PRESETS,
    DEFAULT_LOCAL_LLM_ROLE_PRESETS,
    ActivityLogConfig,
    ActivityScheduleEntry,
    AnimaDefaults,
    AnimaModelConfig,
    AnimaWorksConfig,
    BackgroundTaskConfig,
    BackgroundToolConfig,
    CommandsPermission,
    ConsolidationConfig,
    CredentialConfig,
    ExternalMessagingChannelConfig,
    ExternalMessagingConfig,
    ExternalToolsPermission,
    GPUConfig,
    HeartbeatConfig,
    HelperModelFallback,
    HelperModelRole,
    HelperModelsConfig,
    HousekeepingConfig,
    HubConfig,
    HumanNotificationConfig,
    ImageGenConfig,
    InboxConfig,
    LocalLLMConfig,
    MediaProxyConfig,
    NotificationChannelConfig,
    PermissionsConfig,
    PhoneConfig,
    PrimingConfig,
    PromptConfig,
    RAGConfig,
    ServerConfig,
    SkillCronConfig,
    SkillPromotionConfig,
    SkillsConfig,
    SystemConfig,
    ToolCreationPermission,
    UIConfig,
    UserAliasConfig,
    VoiceConfig,
    _format_permissions_for_prompt,
    load_permissions,
    resolve_background_worker_pool_size,
    resolve_chat_max_concurrency,
)
