<!-- 自動生成ファイル・編集禁止。再生成: uv run python scripts/gen_reference.py api -->
<!-- generator: gen_reference/1  kind: api  source-sha256: a12fe2b105f9e90e2eb7d2c98520407770fb7e6d0c770e83aea4e13fe994a7ca -->

# API リファレンス

FastAPI の OpenAPI 定義、WebSocket、`server/app.py` の直書きルートから生成しています。

| メソッド | パス | 認証区分 | 概要 | 定義位置 |
|---|---|---|---|---|

## `server/app.py`

| GET | `/` | 不要 | — | `server/app.py:_serve_index` |
| GET | `/_v/{version}/{path:path}` | 不要 | — | `server/app.py:_serve_versioned_static` |
| GET | `/api/workspace/pixel/assets/{path:path}` | セッション必須（local_trust モード、または localhost 信頼が有効なら省略可） | — | `server/app.py:_pixel_workspace_asset` |
| GET | `/api/workspace/pixel/scene` | セッション必須（local_trust モード、または localhost 信頼が有効なら省略可） | — | `server/app.py:_pixel_workspace_scene` |
| GET | `/battle` | 不要 | — | `server/app.py:_serve_battle_index` |
| GET | `/battle/` | 不要 | — | `server/app.py:_serve_battle_index` |
| GET | `/health` | 不要（除外一覧） | — | `server/app.py:_health` |
| GET | `/setup` | 不要 | — | `server/app.py:_serve_setup_index` |
| GET | `/setup/` | 不要 | — | `server/app.py:_serve_setup_index` |
| GET | `/startup-status` | 不要 | — | `server/app.py:_startup_status` |
| GET | `/workspace` | 不要 | — | `server/app.py:_serve_workspace_index` |
| GET | `/workspace/` | 不要 | — | `server/app.py:_serve_workspace_index` |
| GET | `/workspace/pixel` | 不要 | — | `server/app.py:_redirect_pixel_workspace` |
| GET | `/workspace/pixel/` | 不要 | — | `server/app.py:_serve_pixel_workspace_index` |

## `server/routes/animas.py`

| GET | `/api/animas` | セッション必須（local_trust モード、または localhost 信頼が有効なら省略可） | 利用可能な anima の一覧を取得します。 | `server/routes/animas.py:list_animas` |
| POST | `/api/animas/reload-all` | セッション必須（local_trust モード、または localhost 信頼が有効なら省略可） | Hot-reload ModelConfig for all running animas. | `server/routes/animas.py:reload_all_anima_configs` |
| DELETE | `/api/animas/{name}` | セッション必須（local_trust モード、または localhost 信頼が有効なら省略可） | Stop and delete an anima entirely (process + files). | `server/routes/animas.py:delete_anima` |
| GET | `/api/animas/{name}` | セッション必須（local_trust モード、または localhost 信頼が有効なら省略可） | — | `server/routes/animas.py:get_anima_detail` |
| GET | `/api/animas/{name}/aliases` | セッション必須（local_trust モード、または localhost 信頼が有効なら省略可） | Return aliases for an anima from config.json. | `server/routes/animas.py:get_anima_aliases` |
| PUT | `/api/animas/{name}/aliases` | セッション必須（local_trust モード、または localhost 信頼が有効なら省略可） | Update aliases for an anima in config.json. | `server/routes/animas.py:update_anima_aliases` |
| PUT | `/api/animas/{name}/background-model` | セッション必須（local_trust モード、または localhost 信頼が有効なら省略可） | Update heartbeat/cron model settings in root-owned status.json. | `server/routes/animas.py:update_anima_background_model` |
| GET | `/api/animas/{name}/background-tasks` | セッション必須（local_trust モード、または localhost 信頼が有効なら省略可） | List background tasks for an anima (reads from state dir). | `server/routes/animas.py:list_background_tasks` |
| GET | `/api/animas/{name}/background-tasks/{task_id}` | セッション必須（local_trust モード、または localhost 信頼が有効なら省略可） | Get a specific background task by ID. | `server/routes/animas.py:get_background_task` |
| POST | `/api/animas/{name}/bootstrap/repair` | セッション必須（local_trust モード、または localhost 信頼が有効なら省略可） | Recover an Anima whose first-run setup stopped in needs_repair. | `server/routes/animas.py:repair_anima_bootstrap` |
| GET | `/api/animas/{name}/config` | セッション必須（local_trust モード、または localhost 信頼が有効なら省略可） | Return resolved model configuration for an anima. | `server/routes/animas.py:get_anima_config` |
| GET | `/api/animas/{name}/cron` | セッション必須（local_trust モード、または localhost 信頼が有効なら省略可） | Return the raw cron.md content for an anima. | `server/routes/animas.py:get_anima_cron` |
| POST | `/api/animas/{name}/disable` | セッション必須（local_trust モード、または localhost 信頼が有効なら省略可） | Disable an Anima (set status.json to enabled: false and stop process). | `server/routes/animas.py:disable_anima` |
| POST | `/api/animas/{name}/enable` | セッション必須（local_trust モード、または localhost 信頼が有効なら省略可） | Enable an Anima (set status.json to enabled: true). | `server/routes/animas.py:enable_anima` |
| GET | `/api/animas/{name}/heartbeat` | セッション必須（local_trust モード、または localhost 信頼が有効なら省略可） | Return the raw HEARTBEAT.md content for an anima. | `server/routes/animas.py:get_anima_heartbeat` |
| PUT | `/api/animas/{name}/identity` | セッション必須（local_trust モード、または localhost 信頼が有効なら省略可） | Update the IDENTITY.md content for an anima. | `server/routes/animas.py:update_anima_identity` |
| PUT | `/api/animas/{name}/injection` | セッション必須（local_trust モード、または localhost 信頼が有効なら省略可） | Update the AGENTS.md content for an anima. | `server/routes/animas.py:update_anima_injection` |
| POST | `/api/animas/{name}/interactions/{callback_id}/resolve` | セッション必須（local_trust モード、または localhost 信頼が有効なら省略可） | Resolve an interactive call_human request from the main chat UI. | `server/routes/animas.py:resolve_interaction` |
| POST | `/api/animas/{name}/interrupt` | セッション必須（local_trust モード、または localhost 信頼が有効なら省略可） | Interrupt the current LLM session without stopping the process. | `server/routes/animas.py:interrupt_anima` |
| PUT | `/api/animas/{name}/model` | セッション必須（local_trust モード、または localhost 信頼が有効なら省略可） | Update the model setting in status.json for an anima. | `server/routes/animas.py:update_anima_model` |
| GET | `/api/animas/{name}/permissions` | セッション必須（local_trust モード、または localhost 信頼が有効なら省略可） | Return permissions.json for an anima. | `server/routes/animas.py:get_anima_permissions` |
| PUT | `/api/animas/{name}/permissions` | セッション必須（local_trust モード、または localhost 信頼が有効なら省略可） | Update permissions.json for an anima. | `server/routes/animas.py:update_anima_permissions` |
| POST | `/api/animas/{name}/reload` | セッション必須（local_trust モード、または localhost 信頼が有効なら省略可） | Hot-reload ModelConfig from status.json without process restart. | `server/routes/animas.py:reload_anima_config` |
| POST | `/api/animas/{name}/rename` | セッション必須（local_trust モード、または localhost 信頼が有効なら省略可） | Rename an Anima on root so the CLI never moves owned settings itself. | `server/routes/animas.py:rename_anima` |
| POST | `/api/animas/{name}/restart` | セッション必須（local_trust モード、または localhost 信頼が有効なら省略可） | Restart a specific anima process. | `server/routes/animas.py:restart_anima` |
| PUT | `/api/animas/{name}/role` | セッション必須（local_trust モード、または localhost 信頼が有効なら省略可） | Update role settings and role permission template through root writers. | `server/routes/animas.py:update_anima_role` |
| POST | `/api/animas/{name}/start` | セッション必須（local_trust モード、または localhost 信頼が有効なら省略可） | Start a stopped anima process. | `server/routes/animas.py:start_anima` |
| GET | `/api/animas/{name}/state` | セッション必須（local_trust モード、または localhost 信頼が有効なら省略可） | Read only the current-state document for the chat information pane. | `server/routes/animas.py:get_anima_state` |
| POST | `/api/animas/{name}/stop` | セッション必須（local_trust モード、または localhost 信頼が有効なら省略可） | Stop a specific anima process. | `server/routes/animas.py:stop_anima` |
| GET | `/api/animas/{name}/summary` | セッション必須（local_trust モード、または localhost 信頼が有効なら省略可） | Return only the small set of Anima fields needed by chat selection. | `server/routes/animas.py:get_anima_summary` |
| POST | `/api/animas/{name}/trigger` | セッション必須（local_trust モード、または localhost 信頼が有効なら省略可） | — | `server/routes/animas.py:trigger_heartbeat` |
| GET | `/api/org/chart` | セッション必須（local_trust モード、または localhost 信頼が有効なら省略可） | Return the organisation chart as a tree-structured JSON. | `server/routes/animas.py:get_org_chart` |

## `server/routes/approve.py`

| GET | `/api/approve/{callback_id}` | 不要（除外一覧） | Serve the approval page for a given callback_id. | `server/routes/approve.py:get_approval_page` |
| POST | `/api/approve/{callback_id}` | 不要（除外一覧） | Process an approval decision from the web page. | `server/routes/approve.py:submit_approval` |

## `server/routes/assets.py`

| GET | `/api/animas/{name}/assets` | セッション必須（local_trust モード、または localhost 信頼が有効なら省略可） | List available assets for an anima. | `server/routes/assets.py:list_assets` |
| GET | `/api/animas/{name}/assets/face/{emotion}` | セッション必須（local_trust モード、または localhost 信頼が有効なら省略可） | Serve a cached square crop of an Anima's bustup expression. | `server/routes/assets.py:get_face_asset` |
| POST | `/api/animas/{name}/assets/generate` | セッション必須（local_trust モード、または localhost 信頼が有効なら省略可） | Trigger character asset generation pipeline. | `server/routes/assets.py:generate_assets` |
| POST | `/api/animas/{name}/assets/generate-expression` | セッション必須（local_trust モード、または localhost 信頼が有効なら省略可） | Generate a specific bustup expression variant on demand. | `server/routes/assets.py:generate_expression_on_demand` |
| GET | `/api/animas/{name}/assets/metadata` | セッション必須（local_trust モード、または localhost 信頼が有効なら省略可） | Return structured metadata about a anima's available assets. | `server/routes/assets.py:get_asset_metadata` |
| POST | `/api/animas/{name}/assets/regenerate-step` | セッション必須（local_trust モード、または localhost 信頼が有効なら省略可） | Regenerate one pipeline step (fullbody, bustup, icon, chibi, pixel, 3d, rigging, animations). | `server/routes/assets.py:regenerate_asset_step` |
| POST | `/api/animas/{name}/assets/remake-confirm` | セッション必須（local_trust モード、または localhost 信頼が有効なら省略可） | Accept the preview and cascade-rebuild all remaining assets. | `server/routes/assets.py:remake_confirm` |
| DELETE | `/api/animas/{name}/assets/remake-preview` | セッション必須（local_trust モード、または localhost 信頼が有効なら省略可） | Cancel a remake preview by restoring from the most recent backup. | `server/routes/assets.py:cancel_remake_preview` |
| GET | `/api/animas/{name}/assets/remake-preview` | セッション必須（local_trust モード、または localhost 信頼が有効なら省略可） | List generated preview files. | `server/routes/assets.py:list_remake_previews` |
| POST | `/api/animas/{name}/assets/remake-preview` | セッション必須（local_trust モード、または localhost 信頼が有効なら省略可） | Generate a fullbody preview, optionally using Vibe Transfer. | `server/routes/assets.py:remake_preview` |
| POST | `/api/animas/{name}/assets/upload-fullbody` | セッション必須（local_trust モード、または localhost 信頼が有効なら省略可） | Upload a PNG/JPEG to overwrite the anima's full-body reference image. | `server/routes/assets.py:upload_fullbody_asset` |
| GET | `/api/animas/{name}/assets/{filename}` | セッション必須（local_trust モード、または localhost 信頼が有効なら省略可） | Serve a static asset file from a anima's assets directory. | `server/routes/assets.py:get_asset` |
| HEAD | `/api/animas/{name}/assets/{filename}` | セッション必須（local_trust モード、または localhost 信頼が有効なら省略可） | Serve a static asset file from a anima's assets directory. | `server/routes/assets.py:get_asset` |
| GET | `/api/animas/{name}/attachments/{filename}` | セッション必須（local_trust モード、または localhost 信頼が有効なら省略可） | Serve a user-uploaded attachment from a anima's attachments directory. | `server/routes/assets.py:get_attachment` |
| HEAD | `/api/animas/{name}/attachments/{filename}` | セッション必須（local_trust モード、または localhost 信頼が有効なら省略可） | Serve a user-uploaded attachment from a anima's attachments directory. | `server/routes/assets.py:get_attachment` |
| GET | `/api/animas/{name}/attachments/{filename}/preview` | セッション必須（local_trust モード、または localhost 信頼が有効なら省略可） | Return a structure-only JSON preview of a chat attachment. | `server/routes/assets.py:get_attachment_preview` |
| GET | `/api/media/proxy` | セッション必須（local_trust モード、または localhost 信頼が有効なら省略可） | — | `server/routes/assets.py:media_proxy` |

## `server/routes/auth.py`

| POST | `/api/auth/login` | 不要（除外一覧） | 認証情報を検証し、セッションを開始します。 | `server/routes/auth.py:login` |
| POST | `/api/auth/logout` | セッション必須（local_trust モード、または localhost 信頼が有効なら省略可） | — | `server/routes/auth.py:logout` |
| GET | `/api/auth/me` | セッション必須（local_trust モード、または localhost 信頼が有効なら省略可） | — | `server/routes/auth.py:me` |

## `server/routes/channels.py`

| GET | `/api/channels` | セッション必須（local_trust モード、または localhost 信頼が有効なら省略可） | List all shared channels with metadata including ACL info. | `server/routes/channels.py:list_channels` |
| GET | `/api/channels/{name}` | セッション必須（local_trust モード、または localhost 信頼が有効なら省略可） | Get messages from a specific channel. | `server/routes/channels.py:get_channel_messages` |
| POST | `/api/channels/{name}` | セッション必須（local_trust モード、または localhost 信頼が有効なら省略可） | Post a message to a channel (human-originated). | `server/routes/channels.py:post_to_channel` |
| GET | `/api/channels/{name}/mentions/{anima}` | セッション必須（local_trust モード、または localhost 信頼が有効なら省略可） | Get messages mentioning a specific anima in a channel. | `server/routes/channels.py:get_channel_mentions` |
| GET | `/api/dm` | セッション必須（local_trust モード、または localhost 信頼が有効なら省略可） | List all DM conversation pairs with metadata. | `server/routes/channels.py:list_dm_pairs` |
| GET | `/api/dm/{pair}` | セッション必須（local_trust モード、または localhost 信頼が有効なら省略可） | Get DM history for a specific pair. | `server/routes/channels.py:get_dm_history` |

## `server/routes/chat.py`

| POST | `/api/animas/{name}/chat` | セッション必須（local_trust モード、または localhost 信頼が有効なら省略可） | — | `server/routes/chat.py:chat` |
| POST | `/api/animas/{name}/chat/compact` | セッション必須（local_trust モード、または localhost 信頼が有効なら省略可） | Manually compact a chat thread's context on demand. | `server/routes/chat.py:compact_session` |
| POST | `/api/animas/{name}/chat/stream` | セッション必須（local_trust モード、または localhost 信頼が有効なら省略可） | Stream chat response via SSE over IPC. | `server/routes/chat.py:chat_stream` |
| POST | `/api/animas/{name}/greet` | セッション必須（local_trust モード、または localhost 信頼が有効なら省略可） | Generate a greeting when user clicks the character. | `server/routes/chat.py:greet` |
| GET | `/api/animas/{name}/stream/active` | セッション必須（local_trust モード、または localhost 信頼が有効なら省略可） | Return the active (or most recent) stream for an anima. | `server/routes/chat.py:get_active_stream` |
| GET | `/api/animas/{name}/stream/{response_id}/progress` | セッション必須（local_trust モード、または localhost 信頼が有効なら省略可） | Return progress of a specific stream. | `server/routes/chat.py:get_stream_progress` |

## `server/routes/chat_ui_state.py`

| GET | `/api/chat/ui-state` | セッション必須（local_trust モード、または localhost 信頼が有効なら省略可） | Get persisted dashboard chat UI state for current user. | `server/routes/chat_ui_state.py:get_chat_ui_state` |
| PUT | `/api/chat/ui-state` | セッション必須（local_trust モード、または localhost 信頼が有効なら省略可） | Persist dashboard chat UI state for current user. | `server/routes/chat_ui_state.py:put_chat_ui_state` |

## `server/routes/config_routes.py`

| GET | `/api/discord/channel-members` | セッション必須（local_trust モード、または localhost 信頼が有効なら省略可） | Return all Discord channel membership mappings. | `server/routes/config_routes.py:get_discord_channel_members` |
| PUT | `/api/discord/channel-members/{channel_id}` | セッション必須（local_trust モード、または localhost 信頼が有効なら省略可） | Update Anima members for a Discord channel. | `server/routes/config_routes.py:put_discord_channel_members` |
| GET | `/api/discord/channels` | セッション必須（local_trust モード、または localhost 信頼が有効なら省略可） | List Discord guild channels with membership info. | `server/routes/config_routes.py:get_discord_channels` |
| GET | `/api/settings/anthropic-auth` | セッション必須（local_trust モード、または localhost 信頼が有効なら省略可） | Return current Anthropic auth mode and runtime availability. | `server/routes/config_routes.py:get_anthropic_auth` |
| PUT | `/api/settings/anthropic-auth` | セッション必須（local_trust モード、または localhost 信頼が有効なら省略可） | Persist Anthropic auth mode in config.json for the settings UI. | `server/routes/config_routes.py:update_anthropic_auth` |
| GET | `/api/settings/openai-auth` | セッション必須（local_trust モード、または localhost 信頼が有効なら省略可） | Return current OpenAI auth mode and runtime availability. | `server/routes/config_routes.py:get_openai_auth` |
| PUT | `/api/settings/openai-auth` | セッション必須（local_trust モード、または localhost 信頼が有効なら省略可） | Persist OpenAI auth mode in config.json for the settings UI. | `server/routes/config_routes.py:update_openai_auth` |
| GET | `/api/system/available-models` | セッション必須（local_trust モード、または localhost 信頼が有効なら省略可） | Return all available models (cloud + local) for UI dropdowns. | `server/routes/config_routes.py:get_available_models` |
| GET | `/api/system/available-tools` | セッション必須（local_trust モード、または localhost 信頼が有効なら省略可） | Return available external tool module names (minus disabled services). | `server/routes/config_routes.py:get_available_tools` |
| GET | `/api/system/config` | セッション必須（local_trust モード、または localhost 信頼が有効なら省略可） | Read and return the AnimaWorks config with masked secrets. | `server/routes/config_routes.py:get_config` |
| PUT | `/api/system/config/value` | セッション必須（local_trust モード、または localhost 信頼が有効なら省略可） | Apply one validated CLI config-set operation through the root server. | `server/routes/config_routes.py:update_config_value` |
| PUT | `/api/system/config/wizard` | セッション必須（local_trust モード、または localhost 信頼が有効なら省略可） | Apply the interactive CLI wizard changes on the root server. | `server/routes/config_routes.py:save_config_wizard` |

## `server/routes/external_tasks.py`

| GET | `/api/external-tasks` | セッション必須（local_trust モード、または localhost 信頼が有効なら省略可） | Get external tasks for the widget from the snapshot store. | `server/routes/external_tasks.py:get_external_tasks` |

## `server/routes/internal.py`

| POST | `/api/internal/anima/create` | 内部 | Create an anima outside sandbox EROFS constraints. | `server/routes/internal.py:internal_anima_create` |
| POST | `/api/internal/animas/{target}/control` | 内部 | Persist subordinate control settings after verifying the token owner. | `server/routes/internal.py:internal_anima_control` |
| POST | `/api/internal/animas/{target}/prompt-settings` | 内部 | Apply an authorized identity/injection change through the root owner. | `server/routes/internal.py:internal_update_anima_prompt_setting` |
| POST | `/api/internal/call-human/confirm` | 内部 | Check the CLI ``call_human`` confirmation key; keys live in server memory only. | `server/routes/internal.py:internal_call_human_confirm` |
| POST | `/api/internal/company/assign` | 内部 | Apply CLI company assignments through the root-owned status writer. | `server/routes/internal.py:internal_company_assign` |
| GET | `/api/internal/company/boundary` | 内部 | Resolve company membership on the host for sandboxed handlers. | `server/routes/internal.py:internal_company_boundary` |
| POST | `/api/internal/company/split` | 内部 | Run a CLI company split on root when it mutates status/settings. | `server/routes/internal.py:internal_company_split` |
| POST | `/api/internal/delegate-task` | 内部 | Persist a delegated task outside sandbox EROFS constraints. | `server/routes/internal.py:internal_delegate_task` |
| POST | `/api/internal/embed` | 内部 | Centralized embedding inference for child processes. | `server/routes/internal.py:internal_embed` |
| POST | `/api/internal/interaction/create` | 内部 | — | `server/routes/internal.py:internal_interaction_create` |
| POST | `/api/internal/interaction/message-ts` | 内部 | — | `server/routes/internal.py:internal_interaction_message_ts` |
| POST | `/api/internal/message-sent` | 内部 | Notify the server that a message was sent via CLI. | `server/routes/internal.py:internal_message_sent` |
| POST | `/api/internal/notification-mapping` | 内部 | — | `server/routes/internal.py:internal_notification_mapping` |
| POST | `/api/internal/notify-web` | 内部 | Push a call_human notification into connected Web UI clients. | `server/routes/internal.py:internal_notify_web` |
| POST | `/api/internal/phone/alert` | 内部 | Start an urgent phone alert for the configured Anima. | `server/routes/internal.py:internal_phone_alert` |
| POST | `/api/internal/post-channel` | 内部 | Append a channel post outside sandbox EROFS constraints. | `server/routes/internal.py:internal_post_channel` |
| POST | `/api/internal/rerank` | 内部 | Centralized cross-encoder reranking for child processes. | `server/routes/internal.py:internal_rerank` |
| POST | `/api/internal/send-message` | 内部 | Persist a DM outside sandbox EROFS constraints. | `server/routes/internal.py:internal_send_message` |
| POST | `/api/internal/settings/anima-icon-template` | 内部 | Persist icon-template defaults on behalf of an authenticated worker. | `server/routes/internal.py:internal_persist_anima_icon_template` |
| POST | `/api/internal/submit-tasks` | 内部 | Publish a complete batch on the host; no sandbox DB grant is needed. | `server/routes/internal.py:internal_submit_tasks` |
| POST | `/api/internal/task-board-action` | 内部 | Run a lease-guarded task board write for a sandboxed anima CLI. | `server/routes/internal.py:internal_task_board_action` |
| GET | `/api/internal/tasks` | 内部 | Read a task snapshot for workers without direct database access. | `server/routes/internal.py:internal_tasks` |
| POST | `/api/internal/update-task` | 内部 | Persist a task update outside sandbox EROFS constraints. | `server/routes/internal.py:internal_update_task` |
| POST | `/api/internal/vector/count` | 内部 | — | `server/routes/internal.py:vector_count` |
| POST | `/api/internal/vector/create-collection` | 内部 | — | `server/routes/internal.py:vector_create_collection` |
| POST | `/api/internal/vector/delete-collection` | 内部 | — | `server/routes/internal.py:vector_delete_collection` |
| POST | `/api/internal/vector/delete-documents` | 内部 | — | `server/routes/internal.py:vector_delete_documents` |
| POST | `/api/internal/vector/get-all` | 内部 | — | `server/routes/internal.py:vector_get_all` |
| POST | `/api/internal/vector/get-by-ids` | 内部 | — | `server/routes/internal.py:vector_get_by_ids` |
| POST | `/api/internal/vector/get-by-metadata` | 内部 | — | `server/routes/internal.py:vector_get_by_metadata` |
| POST | `/api/internal/vector/list-collections` | 内部 | — | `server/routes/internal.py:vector_list_collections` |
| POST | `/api/internal/vector/query` | 内部 | 内部サービス向けのベクトル検索を実行します。 | `server/routes/internal.py:vector_query` |
| POST | `/api/internal/vector/update-metadata` | 内部 | — | `server/routes/internal.py:vector_update_metadata` |
| POST | `/api/internal/vector/upsert` | 内部 | — | `server/routes/internal.py:vector_upsert` |
| POST | `/api/internal/workspace/grant` | 内部 | Apply a human-origin workspace grant through root-owned writers. | `server/routes/internal.py:internal_workspace_grant` |
| GET | `/api/messages/{message_id}` | セッション必須（local_trust モード、または localhost 信頼が有効なら省略可） | Return the full JSON of a stored message by its ID. | `server/routes/internal.py:get_message` |

## `server/routes/logs_routes.py`

| GET | `/api/system/logs` | セッション必須（local_trust モード、または localhost 信頼が有効なら省略可） | List available log files. | `server/routes/logs_routes.py:list_logs` |
| GET | `/api/system/logs/file/read` | セッション必須（local_trust モード、または localhost 信頼が有効なら省略可） | Read log file by basename or relative path from list endpoint. | `server/routes/logs_routes.py:read_log_by_ref` |
| GET | `/api/system/logs/stream` | セッション必須（local_trust モード、または localhost 信頼が有効なら省略可） | SSE endpoint for real-time log streaming (tail -f style). | `server/routes/logs_routes.py:stream_logs` |
| GET | `/api/system/logs/{filename}` | セッション必須（local_trust モード、または localhost 信頼が有効なら省略可） | Read log file content with pagination. | `server/routes/logs_routes.py:read_log` |

## `server/routes/memory_routes.py`

| DELETE | `/api/animas/{name}/conversation` | セッション必須（local_trust モード、または localhost 信頼が有効なら省略可） | Clear conversation history for a fresh start. | `server/routes/memory_routes.py:clear_conversation` |
| GET | `/api/animas/{name}/conversation` | セッション必須（local_trust モード、または localhost 信頼が有効なら省略可） | View current conversation state. | `server/routes/memory_routes.py:get_conversation` |
| POST | `/api/animas/{name}/conversation/compress` | セッション必須（local_trust モード、または localhost 信頼が有効なら省略可） | Manually trigger conversation compression. | `server/routes/memory_routes.py:compress_conversation` |
| GET | `/api/animas/{name}/episodes` | セッション必須（local_trust モード、または localhost 信頼が有効なら省略可） | — | `server/routes/memory_routes.py:list_episodes` |
| GET | `/api/animas/{name}/episodes/calendar` | セッション必須（local_trust モード、または localhost 信頼が有効なら省略可） | Return lightweight episode availability for every day in a month. | `server/routes/memory_routes.py:episode_calendar` |
| GET | `/api/animas/{name}/episodes/{date}` | セッション必須（local_trust モード、または localhost 信頼が有効なら省略可） | — | `server/routes/memory_routes.py:get_episode` |
| GET | `/api/animas/{name}/knowledge` | セッション必須（local_trust モード、または localhost 信頼が有効なら省略可） | — | `server/routes/memory_routes.py:list_knowledge` |
| GET | `/api/animas/{name}/knowledge/{topic}` | セッション必須（local_trust モード、または localhost 信頼が有効なら省略可） | — | `server/routes/memory_routes.py:get_knowledge` |
| GET | `/api/animas/{name}/memory/connections` | セッション必須（local_trust モード、または localhost 信頼が有効なら省略可） | Return entity connections extracted from active facts. | `server/routes/memory_routes.py:memory_connections` |
| GET | `/api/animas/{name}/memory/connections/entity` | セッション必須（local_trust モード、または localhost 信頼が有効なら省略可） | Return facts and related memory files for a single entity. | `server/routes/memory_routes.py:memory_connection_entity` |
| GET | `/api/animas/{name}/memory/graph` | セッション必須（local_trust モード、または localhost 信頼が有効なら省略可） | Return the explicit-link memory graph for UI display. | `server/routes/memory_routes.py:memory_graph` |
| GET | `/api/animas/{name}/memory/stats` | セッション必須（local_trust モード、または localhost 信頼が有効なら省略可） | Return memory storage statistics for an anima. | `server/routes/memory_routes.py:memory_stats` |
| GET | `/api/animas/{name}/procedures` | セッション必須（local_trust モード、または localhost 信頼が有効なら省略可） | — | `server/routes/memory_routes.py:list_procedures` |
| GET | `/api/animas/{name}/procedures/{proc}` | セッション必須（local_trust モード、または localhost 信頼が有効なら省略可） | — | `server/routes/memory_routes.py:get_procedure` |

## `server/routes/phone.py`

| GET | `/api/webhooks/twilio/audio/{token}.wav` | 不要（除外一覧） | Serve an unexpired synthesized WAV by its unguessable token. | `server/routes/phone.py:audio` |
| POST | `/api/webhooks/twilio/pin` | 不要（除外一覧） | Validate the PIN and issue a one-use credential for the phone stream. | `server/routes/phone.py:pin` |
| POST | `/api/webhooks/twilio/status` | 不要（除外一覧） | Handle Twilio call status callbacks and release terminal sessions. | `server/routes/phone.py:status` |
| POST | `/api/webhooks/twilio/voice` | 不要（除外一覧） | Handle inbound calls, alert playback, and alert DTMF responses. | `server/routes/phone.py:voice` |

## `server/routes/phone_stream.py`

| WS | `/api/webhooks/twilio/stream` | 不要（除外一覧） | — | `server/routes/phone_stream.py:phone_media_stream` |

## `server/routes/room.py`

| GET | `/api/rooms` | セッション必須（local_trust モード、または localhost 信頼が有効なら省略可） | List meeting rooms. | `server/routes/room.py:list_rooms` |
| POST | `/api/rooms` | セッション必須（local_trust モード、または localhost 信頼が有効なら省略可） | Create a new meeting room. | `server/routes/room.py:create_room` |
| GET | `/api/rooms/{room_id}` | セッション必須（local_trust モード、または localhost 信頼が有効なら省略可） | Get room details including conversation. | `server/routes/room.py:get_room` |
| POST | `/api/rooms/{room_id}/chat/stream` | セッション必須（local_trust モード、または localhost 信頼が有効なら省略可） | Main SSE streaming endpoint for meeting chat. | `server/routes/room.py:meeting_chat_stream` |
| POST | `/api/rooms/{room_id}/close` | セッション必須（local_trust モード、または localhost 信頼が有効なら省略可） | Close room and generate minutes. | `server/routes/room.py:close_room` |
| POST | `/api/rooms/{room_id}/participants` | セッション必須（local_trust モード、または localhost 信頼が有効なら省略可） | Add a participant to the room. | `server/routes/room.py:add_participant` |
| DELETE | `/api/rooms/{room_id}/participants/{name}` | セッション必須（local_trust モード、または localhost 信頼が有効なら省略可） | Remove a participant from the room. | `server/routes/room.py:remove_participant` |

## `server/routes/sessions.py`

| GET | `/api/animas/{name}/conversation/bg` | セッション必須（local_trust モード、または localhost 信頼が有効なら省略可） | Load background activity for one collapsed feed group. | `server/routes/sessions.py:get_conversation_background` |
| POST | `/api/animas/{name}/conversation/calls` | セッション必須（local_trust モード、または localhost 信頼が有効なら省略可） | Persist a completed voice call in the Anima's conversation history. | `server/routes/sessions.py:record_voice_call` |
| GET | `/api/animas/{name}/conversation/feed` | セッション必須（local_trust モード、または localhost 信頼が有効なら省略可） | Get user-centered chat history with lazy background activity groups. | `server/routes/sessions.py:get_conversation_feed` |
| GET | `/api/animas/{name}/conversation/history` | セッション必須（local_trust モード、または localhost 信頼が有効なら省略可） | Get conversation history from activity log. | `server/routes/sessions.py:get_conversation_history` |
| GET | `/api/animas/{name}/conversation/tool-calls` | セッション必須（local_trust モード、または localhost 信頼が有効なら省略可） | Get the full, truncated tool-call details for one assistant message. | `server/routes/sessions.py:get_conversation_tool_calls` |
| POST | `/api/animas/{name}/notifications/reply` | セッション必須（local_trust モード、または localhost 信頼が有効なら省略可） | Route a reply to a human_notify entry back to its originating Anima. | `server/routes/sessions.py:reply_to_notification` |
| GET | `/api/animas/{name}/sessions` | セッション必須（local_trust モード、または localhost 信頼が有効なら省略可） | List all available sessions: active conversation, archives, episodes. | `server/routes/sessions.py:list_sessions` |
| GET | `/api/animas/{name}/sessions/{session_id}` | セッション必須（local_trust モード、または localhost 信頼が有効なら省略可） | Get archived session detail. | `server/routes/sessions.py:get_session_detail` |
| GET | `/api/animas/{name}/transcripts/{date}` | セッション必須（local_trust モード、または localhost 信頼が有効なら省略可） | Get full conversation transcript for a specific date. | `server/routes/sessions.py:get_transcript` |
| GET | `/api/notifications/awaiting` | セッション必須（local_trust モード、または localhost 信頼が有効なら省略可） | List call_human notifications from the last 24 hours that still wait for a reply, per Anima. | `server/routes/sessions.py:list_awaiting_notifications` |

## `server/routes/setup.py`

| POST | `/api/setup/codex/device-login` | 不要（除外一覧） | Return browser-based device auth instructions for Codex login. | `server/routes/setup.py:start_codex_device_login` |
| POST | `/api/setup/complete` | 不要（除外一覧） | Finalize setup: save config, create anima, mark complete. | `server/routes/setup.py:complete_setup` |
| GET | `/api/setup/detect-locale` | 不要（除外一覧） | Detect locale from Accept-Language header. | `server/routes/setup.py:detect_locale` |
| GET | `/api/setup/environment` | 不要（除外一覧） | Return environment information for the setup wizard. | `server/routes/setup.py:get_environment` |
| POST | `/api/setup/validate-key` | 不要（除外一覧） | Validate an API key by making a small test request. | `server/routes/setup.py:validate_key` |

## `server/routes/skills.py`

| GET | `/api/animas/{name}/skills` | セッション必須（local_trust モード、または localhost 信頼が有効なら省略可） | List visible skills for an anima and mark thread-local active entries. | `server/routes/skills.py:list_skills` |
| GET | `/api/animas/{name}/skills/active` | セッション必須（local_trust モード、または localhost 信頼が有効なら省略可） | Return the active skills currently configured for a chat thread. | `server/routes/skills.py:get_active_skills` |
| PUT | `/api/animas/{name}/skills/active` | セッション必須（local_trust モード、または localhost 信頼が有効なら省略可） | Replace active skills for a chat thread. | `server/routes/skills.py:update_active_skills` |
| POST | `/api/animas/{name}/skills/trust` | セッション必須（local_trust モード、または localhost 信頼が有効なら省略可） | Promote a safe skill to trusted operating guidance. | `server/routes/skills.py:trust_skill` |

## `server/routes/system.py`

| GET | `/api/activity/group` | セッション必須（local_trust モード、または localhost 信頼が有効なら省略可） | Return one complete trigger-based activity group by stable ID. | `server/routes/system.py:get_activity_group` |
| GET | `/api/activity/recent` | セッション必須（local_trust モード、または localhost 信頼が有効なら省略可） | Return recent activity events from unified ActivityLogger. | `server/routes/system.py:get_recent_activity` |
| GET | `/api/activity/running-tasks` | セッション必須（local_trust モード、または localhost 信頼が有効なら省略可） | Return active TaskExec workers and other busy lanes grouped by Anima. | `server/routes/system.py:get_running_activity_tasks` |
| GET | `/api/settings/activity-level` | セッション必須（local_trust モード、または localhost 信頼が有効なら省略可） | Return the current global activity level and schedule. | `server/routes/system.py:get_activity_level` |
| PUT | `/api/settings/activity-level` | セッション必須（local_trust モード、または localhost 信頼が有効なら省略可） | Update global activity level and reschedule all heartbeats. | `server/routes/system.py:set_activity_level` |
| PUT | `/api/settings/activity-schedule` | セッション必須（local_trust モード、または localhost 信頼が有効なら省略可） | Update the time-based activity schedule (night mode). | `server/routes/system.py:set_activity_schedule` |
| POST | `/api/settings/display-mode` | セッション必須（local_trust モード、または localhost 信頼が有効なら省略可） | Update display mode and sync config.image_gen.image_style. | `server/routes/system.py:set_display_mode` |
| GET | `/api/shared/users` | セッション必須（local_trust モード、または localhost 信頼が有効なら省略可） | List registered user names from shared/users/. | `server/routes/system.py:list_shared_users` |
| GET | `/api/system/connections` | セッション必須（local_trust モード、または localhost 信頼が有効なら省略可） | Return WebSocket and process connection info. | `server/routes/system.py:system_connections` |
| GET | `/api/system/frontend-logs` | セッション必須（local_trust モード、または localhost 信頼が有効なら省略可） | Read frontend logs from JSONL files with optional filters. | `server/routes/system.py:view_frontend_logs` |
| POST | `/api/system/frontend-logs` | セッション必須（local_trust モード、または localhost 信頼が有効なら省略可） | Receive a batch of frontend log entries and write to daily JSONL. | `server/routes/system.py:receive_frontend_logs` |
| GET | `/api/system/health` | 不要（除外一覧） | Simple health check endpoint. | `server/routes/system.py:health_check` |
| POST | `/api/system/hot-reload/credentials` | セッション必須（local_trust モード、または localhost 信頼が有効なら省略可） | Hot-reload credentials and dependent connections. | `server/routes/system.py:hot_reload_credentials` |
| POST | `/api/system/hot-reload/slack` | セッション必須（local_trust モード、または localhost 信頼が有効なら省略可） | Hot-reload Slack Socket Mode connections only. | `server/routes/system.py:hot_reload_slack` |
| GET | `/api/system/log-level` | セッション必須（local_trust モード、または localhost 信頼が有効なら省略可） | Return the current root log level. | `server/routes/system.py:get_log_level` |
| POST | `/api/system/log-level` | セッション必須（local_trust モード、または localhost 信頼が有効なら省略可） | Change the log level at runtime (no restart required). | `server/routes/system.py:set_log_level` |
| POST | `/api/system/reload` | セッション必須（local_trust モード、または localhost 信頼が有効なら省略可） | Full sync: add new animas, refresh existing, remove deleted. | `server/routes/system.py:reload_animas` |
| POST | `/api/system/rewrite-runtime-refs` | セッション必須（local_trust モード、または localhost 信頼が有効なら省略可） | Synchronize live caches after REWRITE_REFS updates disk state. | `server/routes/system.py:rewrite_runtime_refs` |
| GET | `/api/system/scheduler` | セッション必須（local_trust モード、または localhost 信頼が有効なら省略可） | Return scheduler status and job information. | `server/routes/system.py:system_scheduler` |
| GET | `/api/system/status` | セッション必須（local_trust モード、または localhost 信頼が有効なら省略可） | — | `server/routes/system.py:system_status` |
| GET | `/api/system/token-budget` | セッション必須（local_trust モード、または localhost 信頼が有効なら省略可） | Return current-month token budget status for each Anima. | `server/routes/system.py:get_token_budget` |

## `server/routes/taskboard.py`

| GET | `/api/task-board` | セッション必須（local_trust モード、または localhost 信頼が有効なら省略可） | Return the unified TaskBoard view read straight from the canonical TaskStore. | `server/routes/taskboard.py:list_task_board` |
| GET | `/api/task-board/summary` | セッション必須（local_trust モード、または localhost 信頼が有効なら省略可） | Return TaskBoard summary counts for dashboard use. | `server/routes/taskboard.py:get_task_board_summary` |
| POST | `/api/task-board/{anima_name}/{task_id}/cancel` | セッション必須（local_trust モード、または localhost 信頼が有効なら省略可） | Explicitly cancel a task, overriding any outstanding lease (human operation). | `server/routes/taskboard.py:cancel_task` |

## `server/routes/usage_routes.py`

| GET | `/api/usage` | セッション必須（local_trust モード、または localhost 信頼が有効なら省略可） | Return combined Claude + OpenAI + nanoGPT usage data. | `server/routes/usage_routes.py:get_usage` |
| POST | `/api/usage/claude/relogin` | セッション必須（local_trust モード、または localhost 信頼が有効なら省略可） | — | `server/routes/usage_routes.py:relogin_claude` |
| POST | `/api/usage/openai/relogin` | セッション必須（local_trust モード、または localhost 信頼が有効なら省略可） | — | `server/routes/usage_routes.py:relogin_openai` |

## `server/routes/users.py`

| GET | `/api/users` | セッション必須（local_trust モード、または localhost 信頼が有効なら省略可） | List all users (without password hashes). | `server/routes/users.py:list_users` |
| POST | `/api/users` | セッション必須（local_trust モード、または localhost 信頼が有効なら省略可） | Add a new user (owner only). | `server/routes/users.py:add_user` |
| PUT | `/api/users/me/password` | セッション必須（local_trust モード、または localhost 信頼が有効なら省略可） | Change (or initially set) the current user's password. | `server/routes/users.py:change_password` |
| DELETE | `/api/users/{username}` | セッション必須（local_trust モード、または localhost 信頼が有効なら省略可） | Delete a user (owner only, cannot delete self). | `server/routes/users.py:delete_user` |

## `server/routes/voice.py`

| WS | `/ws/voice/{name}` | セッション必須（local_trust モード、または localhost 信頼が有効なら省略可） | Voice conversation WebSocket for a specific Anima. | `server/routes/voice.py:voice_websocket` |

## `server/routes/webhooks.py`

| POST | `/api/webhooks/chatwork` | 不要（除外一覧） | Handle Chatwork Webhook notifications. | `server/routes/webhooks.py:chatwork_webhook` |
| POST | `/api/webhooks/github` | 不要（除外一覧） | Authenticate and enqueue a GitHub webhook event. | `server/routes/webhooks.py:github_webhook` |
| POST | `/api/webhooks/slack/events` | 不要（除外一覧） | Handle Slack Event Subscriptions. | `server/routes/webhooks.py:slack_events` |
| POST | `/api/webhooks/zoom` | 不要（除外一覧） | Handle Zoom RTMS webhook notifications. | `server/routes/webhooks.py:zoom_webhook` |
| GET | `/api/webhooks/zoom/oauth-callback` | 不要（除外一覧） | Landing page for the Zoom OAuth redirect. | `server/routes/webhooks.py:zoom_oauth_callback` |

## `server/routes/websocket_route.py`

| WS | `/ws` | セッション必須（local_trust モード、または localhost 信頼が有効なら省略可） | — | `server/routes/websocket_route.py:websocket_endpoint` |
