<!-- 自動翻訳ファイル・編集禁止 (AUTO-TRANSLATED, DO NOT EDIT). 正本: docs/ja/reference/api.md -->
<!-- i18n: source-sha256=c0dec1831bd8ee58a52a34ebe423b741d6b5628e22b3c538070e9d72a0d11be3 generated=2026-11-07 engine=luna model=gpt-6-luna-2026-09-22 translator=2 -->

# API 참조

FastAPI의 OpenAPI 정의, WebSocket 및 `server/app.py`에 직접 작성된 라우트에서 생성됩니다.

| 메서드 | 경로 | 인증 구분 | 개요 | 정의 위치 |
|---|---|---|---|---|

## `server/app.py`

| GET | `/` | 불필요 | — | `server/app.py:_serve_index` |
| GET | `/_v/{version}/{path:path}` | 불필요 | — | `server/app.py:_serve_versioned_static` |
| GET | `/api/workspace/pixel/assets/{path:path}` | 세션 필수(local_trust 모드이거나 localhost 신뢰가 활성화된 경우 생략 가능) | — | `server/app.py:_pixel_workspace_asset` |
| GET | `/api/workspace/pixel/scene` | 세션 필수(local_trust 모드이거나 localhost 신뢰가 활성화된 경우 생략 가능) | — | `server/app.py:_pixel_workspace_scene` |
| GET | `/battle` | 불필요 | — | `server/app.py:_serve_battle_index` |
| GET | `/battle/` | 불필요 | — | `server/app.py:_serve_battle_index` |
| GET | `/health` | 불필요(제외 목록) | — | `server/app.py:_health` |
| GET | `/setup` | 불필요 | — | `server/app.py:_serve_setup_index` |
| GET | `/setup/` | 불필요 | — | `server/app.py:_serve_setup_index` |
| GET | `/startup-status` | 불필요 | — | `server/app.py:_startup_status` |
| GET | `/workspace` | 불필요 | — | `server/app.py:_serve_workspace_index` |
| GET | `/workspace/` | 불필요 | — | `server/app.py:_serve_workspace_index` |
| GET | `/workspace/pixel` | 불필요 | — | `server/app.py:_redirect_pixel_workspace` |
| GET | `/workspace/pixel/` | 불필요 | — | `server/app.py:_serve_pixel_workspace_index` |

## `server/routes/animas.py`

| GET | `/api/animas` | 세션 필수(local_trust 모드이거나 localhost 신뢰가 활성화된 경우 생략 가능) | 사용 가능한 anima 목록을 가져옵니다. | `server/routes/animas.py:list_animas` |
| POST | `/api/animas/reload-all` | 세션 필수(local_trust 모드이거나 localhost 신뢰가 활성화된 경우 생략 가능) | 실행 중인 모든 anima의 ModelConfig를 핫 리로드합니다. | `server/routes/animas.py:reload_all_anima_configs` |
| DELETE | `/api/animas/{name}` | 세션 필수(local_trust 모드이거나 localhost 신뢰가 활성화된 경우 생략 가능) | Anima를 완전히 중지하고 삭제합니다(프로세스 및 파일). | `server/routes/animas.py:delete_anima` |
| GET | `/api/animas/{name}` | 세션 필수(local_trust 모드이거나 localhost 신뢰가 활성화된 경우 생략 가능) | — | `server/routes/animas.py:get_anima_detail` |
| GET | `/api/animas/{name}/aliases` | 세션 필수(local_trust 모드이거나 localhost 신뢰가 활성화된 경우 생략 가능) | config.json에서 anima의 별칭을 반환합니다. | `server/routes/animas.py:get_anima_aliases` |
| PUT | `/api/animas/{name}/aliases` | 세션 필수(local_trust 모드이거나 localhost 신뢰가 활성화된 경우 생략 가능) | config.json에서 anima의 별칭을 업데이트합니다. | `server/routes/animas.py:update_anima_aliases` |
| PUT | `/api/animas/{name}/background-model` | 세션 필수(local_trust 모드이거나 localhost 신뢰가 활성화된 경우 생략 가능) | root 소유의 status.json에서 heartbeat/cron 모델 설정을 업데이트합니다. | `server/routes/animas.py:update_anima_background_model` |
| GET | `/api/animas/{name}/background-tasks` | 세션 필수(local_trust 모드이거나 localhost 신뢰가 활성화된 경우 생략 가능) | anima의 백그라운드 작업 목록을 가져옵니다(state 디렉터리에서 읽음). | `server/routes/animas.py:list_background_tasks` |
| GET | `/api/animas/{name}/background-tasks/{task_id}` | 세션 필수(local_trust 모드이거나 localhost 신뢰가 활성화된 경우 생략 가능) | ID로 특정 백그라운드 작업을 가져옵니다. | `server/routes/animas.py:get_background_task` |
| POST | `/api/animas/{name}/bootstrap/repair` | 세션 필수(local_trust 모드이거나 localhost 신뢰가 활성화된 경우 생략 가능) | 최초 실행 설정이 needs_repair 상태에서 중단된 Anima를 복구합니다. | `server/routes/animas.py:repair_anima_bootstrap` |
| GET | `/api/animas/{name}/config` | 세션 필수(local_trust 모드이거나 localhost 신뢰가 활성화된 경우 생략 가능) | anima에 대해 해석된 모델 구성을 반환합니다. | `server/routes/animas.py:get_anima_config` |
| GET | `/api/animas/{name}/cron` | 세션 필수(local_trust 모드이거나 localhost 신뢰가 활성화된 경우 생략 가능) | anima의 원시 cron.md 콘텐츠를 반환합니다. | `server/routes/animas.py:get_anima_cron` |
| POST | `/api/animas/{name}/disable` | 세션 필수(local_trust 모드이거나 localhost 신뢰가 활성화된 경우 생략 가능) | Anima를 비활성화합니다(status.json을 enabled: false로 설정하고 프로세스를 중지). | `server/routes/animas.py:disable_anima` |
| POST | `/api/animas/{name}/enable` | 세션 필수(local_trust 모드이거나 localhost 신뢰가 활성화된 경우 생략 가능) | Anima를 활성화합니다(status.json을 enabled: true로 설정). | `server/routes/animas.py:enable_anima` |
| GET | `/api/animas/{name}/heartbeat` | 세션 필수(local_trust 모드이거나 localhost 신뢰가 활성화된 경우 생략 가능) | anima의 원시 HEARTBEAT.md 콘텐츠를 반환합니다. | `server/routes/animas.py:get_anima_heartbeat` |
| PUT | `/api/animas/{name}/identity` | 세션 필수(local_trust 모드이거나 localhost 신뢰가 활성화된 경우 생략 가능) | anima의 IDENTITY.md 콘텐츠를 업데이트합니다. | `server/routes/animas.py:update_anima_identity` |
| PUT | `/api/animas/{name}/injection` | 세션 필수(local_trust 모드이거나 localhost 신뢰가 활성화된 경우 생략 가능) | anima의 AGENTS.md 콘텐츠를 업데이트합니다. | `server/routes/animas.py:update_anima_injection` |
| POST | `/api/animas/{name}/interactions/{callback_id}/resolve` | 세션 필수(local_trust 모드이거나 localhost 신뢰가 활성화된 경우 생략 가능) | 기본 채팅 UI에서 대화형 call_human 요청을 처리합니다. | `server/routes/animas.py:resolve_interaction` |
| POST | `/api/animas/{name}/interrupt` | 세션 필수(local_trust 모드이거나 localhost 신뢰가 활성화된 경우 생략 가능) | 프로세스를 중지하지 않고 현재 LLM 세션을 중단합니다. | `server/routes/animas.py:interrupt_anima` |
| PUT | `/api/animas/{name}/model` | 세션 필수(local_trust 모드이거나 localhost 신뢰가 활성화된 경우 생략 가능) | anima의 status.json에 있는 모델 설정을 업데이트합니다. | `server/routes/animas.py:update_anima_model` |
| GET | `/api/animas/{name}/permissions` | 세션 필수(local_trust 모드이거나 localhost 신뢰가 활성화된 경우 생략 가능) | anima의 permissions.json을 반환합니다. | `server/routes/animas.py:get_anima_permissions` |
| PUT | `/api/animas/{name}/permissions` | 세션 필수(local_trust 모드이거나 localhost 신뢰가 활성화된 경우 생략 가능) | anima의 permissions.json을 업데이트합니다. | `server/routes/animas.py:update_anima_permissions` |
| POST | `/api/animas/{name}/reload` | 세션 필수(local_trust 모드이거나 localhost 신뢰가 활성화된 경우 생략 가능) | 프로세스를 재시작하지 않고 status.json에서 ModelConfig를 핫 리로드합니다. | `server/routes/animas.py:reload_anima_config` |
| POST | `/api/animas/{name}/rename` | 세션 필수(local_trust 모드이거나 localhost 신뢰가 활성화된 경우 생략 가능) | CLI가 root 소유 설정을 직접 옮기지 않도록 root에서 Anima의 이름을 변경합니다. | `server/routes/animas.py:rename_anima` |
| POST | `/api/animas/{name}/restart` | 세션 필수(local_trust 모드이거나 localhost 신뢰가 활성화된 경우 생략 가능) | 특정 anima 프로세스를 다시 시작합니다. | `server/routes/animas.py:restart_anima` |
| PUT | `/api/animas/{name}/role` | 세션 필수(local_trust 모드이거나 localhost 신뢰가 활성화된 경우 생략 가능) | root 작성자를 통해 역할 설정 및 역할 권한 템플릿을 업데이트합니다. | `server/routes/animas.py:update_anima_role` |
| POST | `/api/animas/{name}/start` | 세션 필수(local_trust 모드이거나 localhost 신뢰가 활성화된 경우 생략 가능) | 중지된 anima 프로세스를 시작합니다. | `server/routes/animas.py:start_anima` |
| GET | `/api/animas/{name}/state` | 세션 필수(local_trust 모드이거나 localhost 신뢰가 활성화된 경우 생략 가능) | 채팅 정보 창을 위해 현재 상태 문서만 읽습니다. | `server/routes/animas.py:get_anima_state` |
| POST | `/api/animas/{name}/stop` | 세션 필수(local_trust 모드이거나 localhost 신뢰가 활성화된 경우 생략 가능) | 특정 anima 프로세스를 중지합니다. | `server/routes/animas.py:stop_anima` |
| GET | `/api/animas/{name}/summary` | 세션 필수(local_trust 모드이거나 localhost 신뢰가 활성화된 경우 생략 가능) | 채팅 선택에 필요한 소수의 Anima 필드만 반환합니다. | `server/routes/animas.py:get_anima_summary` |
| POST | `/api/animas/{name}/trigger` | 세션 필수(local_trust 모드이거나 localhost 신뢰가 활성화된 경우 생략 가능) | — | `server/routes/animas.py:trigger_heartbeat` |
| GET | `/api/org/chart` | 세션 필수(local_trust 모드이거나 localhost 신뢰가 활성화된 경우 생략 가능) | 조직도를 트리 구조의 JSON으로 반환합니다. | `server/routes/animas.py:get_org_chart` |

## `server/routes/approve.py`

| GET | `/api/approve/{callback_id}` | 불필요(제외 목록) | 지정된 callback_id의 승인 페이지를 제공합니다. | `server/routes/approve.py:get_approval_page` |
| POST | `/api/approve/{callback_id}` | 불필요(제외 목록) | 웹 페이지에서 승인 결정을 처리합니다. | `server/routes/approve.py:submit_approval` |

## `server/routes/assets.py`

| GET | `/api/animas/{name}/assets` | 세션 필수(local_trust 모드 또는 localhost 신뢰가 활성화된 경우 생략 가능) | Anima에서 사용 가능한 에셋을 나열합니다. | `server/routes/assets.py:list_assets` |
| GET | `/api/animas/{name}/assets/face/{emotion}` | 세션 필수(local_trust 모드 또는 localhost 신뢰가 활성화된 경우 생략 가능) | Anima의 상반신 표정 이미지를 캐시된 정사각형 크롭으로 제공합니다. | `server/routes/assets.py:get_face_asset` |
| POST | `/api/animas/{name}/assets/generate` | 세션 필수(local_trust 모드 또는 localhost 신뢰가 활성화된 경우 생략 가능) | 캐릭터 에셋 생성 파이프라인을 시작합니다. | `server/routes/assets.py:generate_assets` |
| POST | `/api/animas/{name}/assets/generate-expression` | 세션 필수(local_trust 모드 또는 localhost 신뢰가 활성화된 경우 생략 가능) | 요청에 따라 특정 상반신 표정 변형을 생성합니다. | `server/routes/assets.py:generate_expression_on_demand` |
| GET | `/api/animas/{name}/assets/metadata` | 세션 필수(local_trust 모드 또는 localhost 신뢰가 활성화된 경우 생략 가능) | Anima에서 사용 가능한 에셋에 관한 구조화된 메타데이터를 반환합니다. | `server/routes/assets.py:get_asset_metadata` |
| POST | `/api/animas/{name}/assets/regenerate-step` | 세션 필수(local_trust 모드 또는 localhost 신뢰가 활성화된 경우 생략 가능) | 파이프라인의 한 단계를 다시 생성합니다(fullbody, bustup, icon, chibi, pixel, 3d, rigging, animations). | `server/routes/assets.py:regenerate_asset_step` |
| POST | `/api/animas/{name}/assets/remake-confirm` | 세션 필수(local_trust 모드 또는 localhost 신뢰가 활성화된 경우 생략 가능) | 미리보기를 승인하고 나머지 모든 에셋을 연쇄적으로 다시 생성합니다. | `server/routes/assets.py:remake_confirm` |
| DELETE | `/api/animas/{name}/assets/remake-preview` | 세션 필수(local_trust 모드 또는 localhost 신뢰가 활성화된 경우 생략 가능) | 가장 최근 백업에서 복원하여 리메이크 미리보기를 취소합니다. | `server/routes/assets.py:cancel_remake_preview` |
| GET | `/api/animas/{name}/assets/remake-preview` | 세션 필수(local_trust 모드 또는 localhost 신뢰가 활성화된 경우 생략 가능) | 생성된 미리보기 파일을 나열합니다. | `server/routes/assets.py:list_remake_previews` |
| POST | `/api/animas/{name}/assets/remake-preview` | 세션 필수(local_trust 모드 또는 localhost 신뢰가 활성화된 경우 생략 가능) | 선택적으로 Vibe Transfer를 사용하여 전신 미리보기를 생성합니다. | `server/routes/assets.py:remake_preview` |
| POST | `/api/animas/{name}/assets/upload-fullbody` | 세션 필수(local_trust 모드 또는 localhost 신뢰가 활성화된 경우 생략 가능) | PNG/JPEG를 업로드하여 Anima의 전신 참조 이미지를 덮어씁니다. | `server/routes/assets.py:upload_fullbody_asset` |
| GET | `/api/animas/{name}/assets/{filename}` | 세션 필수(local_trust 모드 또는 localhost 신뢰가 활성화된 경우 생략 가능) | Anima의 에셋 디렉터리에서 정적 에셋 파일을 제공합니다. | `server/routes/assets.py:get_asset` |
| HEAD | `/api/animas/{name}/assets/{filename}` | 세션 필수(local_trust 모드 또는 localhost 신뢰가 활성화된 경우 생략 가능) | Anima의 에셋 디렉터리에서 정적 에셋 파일을 제공합니다. | `server/routes/assets.py:get_asset` |
| GET | `/api/animas/{name}/attachments/{filename}` | 세션 필수(local_trust 모드 또는 localhost 신뢰가 활성화된 경우 생략 가능) | Anima의 첨부 파일 디렉터리에서 사용자가 업로드한 첨부 파일을 제공합니다. | `server/routes/assets.py:get_attachment` |
| HEAD | `/api/animas/{name}/attachments/{filename}` | 세션 필수(local_trust 모드 또는 localhost 신뢰가 활성화된 경우 생략 가능) | Anima의 첨부 파일 디렉터리에서 사용자가 업로드한 첨부 파일을 제공합니다. | `server/routes/assets.py:get_attachment` |
| GET | `/api/animas/{name}/attachments/{filename}/preview` | 세션 필수(local_trust 모드 또는 localhost 신뢰가 활성화된 경우 생략 가능) | 채팅 첨부 파일의 구조 전용 JSON 미리보기를 반환합니다. | `server/routes/assets.py:get_attachment_preview` |
| GET | `/api/media/proxy` | 세션 필수(local_trust 모드 또는 localhost 신뢰가 활성화된 경우 생략 가능) | — | `server/routes/assets.py:media_proxy` |

## `server/routes/auth.py`

| POST | `/api/auth/login` | 불필요(제외 목록) | 인증 정보를 검증하고 세션을 시작합니다. | `server/routes/auth.py:login` |
| POST | `/api/auth/logout` | 세션 필수(local_trust 모드 또는 localhost 신뢰가 활성화된 경우 생략 가능) | — | `server/routes/auth.py:logout` |
| GET | `/api/auth/me` | 세션 필수(local_trust 모드 또는 localhost 신뢰가 활성화된 경우 생략 가능) | — | `server/routes/auth.py:me` |

## `server/routes/channels.py`

| GET | `/api/channels` | 세션 필수(local_trust 모드 또는 localhost 신뢰가 활성화된 경우 생략 가능) | List all shared channels with metadata including ACL info. | `server/routes/channels.py:list_channels` |
| GET | `/api/channels/{name}` | 세션 필수(local_trust 모드 또는 localhost 신뢰가 활성화된 경우 생략 가능) | Get messages from a specific channel. | `server/routes/channels.py:get_channel_messages` |
| POST | `/api/channels/{name}` | 세션 필수(local_trust 모드 또는 localhost 신뢰가 활성화된 경우 생략 가능) | Post a message to a channel (human-originated). | `server/routes/channels.py:post_to_channel` |
| GET | `/api/channels/{name}/info` | 세션 필수(local_trust 모드 또는 localhost 신뢰가 활성화된 경우 생략 가능) | Return the informational panel for one Board channel. | `server/routes/channels.py:get_channel_info` |
| GET | `/api/channels/{name}/mentions/{anima}` | 세션 필수(local_trust 모드 또는 localhost 신뢰가 활성화된 경우 생략 가능) | Get messages mentioning a specific anima in a channel. | `server/routes/channels.py:get_channel_mentions` |
| POST | `/api/channels/{name}/read` | 세션 필수(local_trust 모드 또는 localhost 신뢰가 활성화된 경우 생략 가능) | Advance the requesting human's stored read position in one channel. | `server/routes/channels.py:mark_channel_read` |
| GET | `/api/channels/{name}/reads` | 세션 필수(local_trust 모드 또는 localhost 신뢰가 활성화된 경우 생략 가능) | Return the stored read positions for one Board channel. | `server/routes/channels.py:get_channel_reads` |
| PUT | `/api/channels/{name}/settings` | 세션 필수(local_trust 모드 또는 localhost 신뢰가 활성화된 경우 생략 가능) | Update a Board's description, visibility, and members (owner/admin only). | `server/routes/channels.py:update_channel_settings` |
| GET | `/api/dm` | 세션 필수(local_trust 모드 또는 localhost 신뢰가 활성화된 경우 생략 가능) | List all DM conversation pairs with metadata. | `server/routes/channels.py:list_dm_pairs` |
| GET | `/api/dm/{pair}` | 세션 필수(local_trust 모드 또는 localhost 신뢰가 활성화된 경우 생략 가능) | Get DM history for a specific pair. | `server/routes/channels.py:get_dm_history` |
| GET | `/api/flows` | 세션 필수(local_trust 모드 또는 localhost 신뢰가 활성화된 경우 생략 가능) | Return Board/DM calls and their summary for the flow view. | `server/routes/channels.py:list_flows` |

## `server/routes/chat.py`

| POST | `/api/animas/{name}/chat` | 세션 필수(local_trust 모드 또는 localhost 신뢰가 활성화된 경우 생략 가능) | — | `server/routes/chat.py:chat` |
| POST | `/api/animas/{name}/chat/compact` | 세션 필수(local_trust 모드 또는 localhost 신뢰가 활성화된 경우 생략 가능) | 요청에 따라 채팅 스레드의 컨텍스트를 수동으로 압축합니다. | `server/routes/chat.py:compact_session` |
| POST | `/api/animas/{name}/chat/stream` | 세션 필수(local_trust 모드 또는 localhost 신뢰가 활성화된 경우 생략 가능) | IPC를 통해 SSE로 채팅 응답을 스트리밍합니다. | `server/routes/chat.py:chat_stream` |
| POST | `/api/animas/{name}/greet` | 세션 필수(local_trust 모드 또는 localhost 신뢰가 활성화된 경우 생략 가능) | 사용자가 캐릭터를 클릭하면 인사말을 생성합니다. | `server/routes/chat.py:greet` |
| GET | `/api/animas/{name}/stream/active` | 세션 필수(local_trust 모드 또는 localhost 신뢰가 활성화된 경우 생략 가능) | Anima의 활성 스트림(또는 가장 최근 스트림)을 반환합니다. | `server/routes/chat.py:get_active_stream` |
| GET | `/api/animas/{name}/stream/{response_id}/progress` | 세션 필수(local_trust 모드 또는 localhost 신뢰가 활성화된 경우 생략 가능) | 특정 스트림의 진행 상태를 반환합니다. | `server/routes/chat.py:get_stream_progress` |

## `server/routes/chat_ui_state.py`

| GET | `/api/chat/ui-state` | 세션 필수(local_trust 모드 또는 localhost 신뢰가 활성화된 경우 생략 가능) | 현재 사용자의 대시보드 채팅 UI 상태를 가져옵니다. | `server/routes/chat_ui_state.py:get_chat_ui_state` |
| PUT | `/api/chat/ui-state` | 세션 필수(local_trust 모드 또는 localhost 신뢰가 활성화된 경우 생략 가능) | 현재 사용자의 대시보드 채팅 UI 상태를 저장합니다. | `server/routes/chat_ui_state.py:put_chat_ui_state` |

## `server/routes/config_routes.py`

| GET | `/api/discord/channel-members` | 세션 필수(local_trust 모드이거나 localhost 신뢰가 활성화된 경우 생략 가능) | 모든 Discord 채널 멤버십 매핑을 반환합니다. | `server/routes/config_routes.py:get_discord_channel_members` |
| PUT | `/api/discord/channel-members/{channel_id}` | 세션 필수(local_trust 모드이거나 localhost 신뢰가 활성화된 경우 생략 가능) | Discord 채널의 Anima 멤버를 업데이트합니다. | `server/routes/config_routes.py:put_discord_channel_members` |
| GET | `/api/discord/channels` | 세션 필수(local_trust 모드이거나 localhost 신뢰가 활성화된 경우 생략 가능) | 멤버십 정보를 포함한 Discord 길드 채널 목록을 반환합니다. | `server/routes/config_routes.py:get_discord_channels` |
| GET | `/api/settings/anthropic-auth` | 세션 필수(local_trust 모드이거나 localhost 신뢰가 활성화된 경우 생략 가능) | 현재 Anthropic 인증 모드와 런타임 사용 가능 여부를 반환합니다. | `server/routes/config_routes.py:get_anthropic_auth` |
| PUT | `/api/settings/anthropic-auth` | 세션 필수(local_trust 모드이거나 localhost 신뢰가 활성화된 경우 생략 가능) | 설정 UI에서 사용할 Anthropic 인증 모드를 config.json에 저장합니다. | `server/routes/config_routes.py:update_anthropic_auth` |
| GET | `/api/settings/openai-auth` | 세션 필수(local_trust 모드이거나 localhost 신뢰가 활성화된 경우 생략 가능) | 현재 OpenAI 인증 모드와 런타임 사용 가능 여부를 반환합니다. | `server/routes/config_routes.py:get_openai_auth` |
| PUT | `/api/settings/openai-auth` | 세션 필수(local_trust 모드이거나 localhost 신뢰가 활성화된 경우 생략 가능) | 설정 UI에서 사용할 OpenAI 인증 모드를 config.json에 저장합니다. | `server/routes/config_routes.py:update_openai_auth` |
| GET | `/api/system/available-models` | 세션 필수(local_trust 모드이거나 localhost 신뢰가 활성화된 경우 생략 가능) | UI 드롭다운에 표시할 사용 가능한 모든 모델(클라우드 + 로컬)을 반환합니다. | `server/routes/config_routes.py:get_available_models` |
| GET | `/api/system/available-tools` | 세션 필수(local_trust 모드이거나 localhost 신뢰가 활성화된 경우 생략 가능) | 사용 가능한 외부 도구 모듈 이름(비활성화된 서비스 제외)을 반환합니다. | `server/routes/config_routes.py:get_available_tools` |
| GET | `/api/system/config` | 세션 필수(local_trust 모드이거나 localhost 신뢰가 활성화된 경우 생략 가능) | AnimaWorks 설정을 읽고 비밀 정보를 마스킹하여 반환합니다. | `server/routes/config_routes.py:get_config` |
| PUT | `/api/system/config/value` | 세션 필수(local_trust 모드이거나 localhost 신뢰가 활성화된 경우 생략 가능) | 루트 서버를 통해 검증된 CLI config-set 작업 하나를 적용합니다. | `server/routes/config_routes.py:update_config_value` |
| PUT | `/api/system/config/wizard` | 세션 필수(local_trust 모드이거나 localhost 신뢰가 활성화된 경우 생략 가능) | 루트 서버에서 대화형 CLI 마법사의 변경 사항을 적용합니다. | `server/routes/config_routes.py:save_config_wizard` |

## `server/routes/external_tasks.py`

| GET | `/api/external-tasks` | 세션 필수(local_trust 모드이거나 localhost 신뢰가 활성화된 경우 생략 가능) | 스냅샷 저장소에서 위젯용 외부 작업을 가져옵니다. | `server/routes/external_tasks.py:get_external_tasks` |

## `server/routes/internal.py`

| POST | `/api/internal/anima/create` | 내부 | 샌드박스의 EROFS 제약을 우회해 anima를 생성합니다. | `server/routes/internal.py:internal_anima_create` |
| POST | `/api/internal/animas/{target}/control` | 내부 | 토큰 소유자를 확인한 후 하위 제어 설정을 저장합니다. | `server/routes/internal.py:internal_anima_control` |
| POST | `/api/internal/animas/{target}/prompt-settings` | 내부 | 루트 소유자를 통해 승인된 IDENTITY/SOUL/AGENTS 변경을 적용합니다. | `server/routes/internal.py:internal_update_anima_prompt_setting` |
| POST | `/api/internal/call-human/confirm` | 내부 | CLI ``call_human`` 확인 키를 검사합니다. 키는 서버 메모리에만 저장됩니다. | `server/routes/internal.py:internal_call_human_confirm` |
| POST | `/api/internal/company/assign` | 내부 | 루트 소유 상태 기록기를 통해 CLI 회사 할당을 적용합니다. | `server/routes/internal.py:internal_company_assign` |
| GET | `/api/internal/company/boundary` | 내부 | 샌드박스 처리된 핸들러를 위해 호스트에서 회사 멤버십을 확인합니다. | `server/routes/internal.py:internal_company_boundary` |
| POST | `/api/internal/company/split` | 내부 | status/settings.을 변경하는 CLI 회사 분할 작업을 루트에서 실행합니다. | `server/routes/internal.py:internal_company_split` |
| POST | `/api/internal/delegate-task` | 내부 | 샌드박스의 EROFS 제약을 우회해 위임된 작업을 저장합니다. | `server/routes/internal.py:internal_delegate_task` |
| POST | `/api/internal/embed` | 내부 | 자식 프로세스를 위한 임베딩 추론을 중앙에서 처리합니다. | `server/routes/internal.py:internal_embed` |
| POST | `/api/internal/interaction/create` | 내부 | — | `server/routes/internal.py:internal_interaction_create` |
| POST | `/api/internal/interaction/message-ts` | 내부 | — | `server/routes/internal.py:internal_interaction_message_ts` |
| POST | `/api/internal/message-sent` | 내부 | CLI를 통해 메시지가 전송되었음을 서버에 알립니다. | `server/routes/internal.py:internal_message_sent` |
| POST | `/api/internal/notification-mapping` | 내부 | — | `server/routes/internal.py:internal_notification_mapping` |
| POST | `/api/internal/notify-web` | 내부 | 연결된 Web UI 클라이언트에 call_human 알림을 전송합니다. | `server/routes/internal.py:internal_notify_web` |
| POST | `/api/internal/phone/alert` | 내부 | 설정된 Anima에 긴급 전화 알림을 시작합니다. | `server/routes/internal.py:internal_phone_alert` |
| POST | `/api/internal/post-channel` | 내부 | 중앙 집중식 Board 기록기를 통해 공유 채널에 게시합니다. | `server/routes/internal.py:internal_post_channel` |
| POST | `/api/internal/rerank` | 내부 | 자식 프로세스를 위한 교차 인코더 재순위를 중앙에서 처리합니다. | `server/routes/internal.py:internal_rerank` |
| POST | `/api/internal/send-message` | 내부 | 샌드박스의 EROFS 제약을 우회해 DM을 저장합니다. | `server/routes/internal.py:internal_send_message` |
| POST | `/api/internal/settings/anima-icon-template` | 내부 | 인증된 작업자를 대신해 아이콘 템플릿 기본값을 저장합니다. | `server/routes/internal.py:internal_persist_anima_icon_template` |
| POST | `/api/internal/submit-tasks` | 내부 | 호스트에서 전체 배치를 게시합니다. 샌드박스 DB 권한은 필요하지 않습니다. | `server/routes/internal.py:internal_submit_tasks` |
| POST | `/api/internal/task-board-action` | 내부 | 샌드박스 anima CLI를 위해 임대 보호가 적용된 작업 보드 기록을 실행합니다. | `server/routes/internal.py:internal_task_board_action` |
| GET | `/api/internal/tasks` | 내부 | 데이터베이스에 직접 접근할 수 없는 작업자를 위해 작업 스냅샷을 읽습니다. | `server/routes/internal.py:internal_tasks` |
| POST | `/api/internal/update-task` | 내부 | 샌드박스의 EROFS 제약을 우회해 작업 업데이트를 저장합니다. | `server/routes/internal.py:internal_update_task` |
| POST | `/api/internal/vector/count` | 내부 | — | `server/routes/internal.py:vector_count` |
| POST | `/api/internal/vector/create-collection` | 내부 | — | `server/routes/internal.py:vector_create_collection` |
| POST | `/api/internal/vector/delete-collection` | 내부 | — | `server/routes/internal.py:vector_delete_collection` |
| POST | `/api/internal/vector/delete-documents` | 내부 | — | `server/routes/internal.py:vector_delete_documents` |
| POST | `/api/internal/vector/get-all` | 내부 | — | `server/routes/internal.py:vector_get_all` |
| POST | `/api/internal/vector/get-by-ids` | 내부 | — | `server/routes/internal.py:vector_get_by_ids` |
| POST | `/api/internal/vector/get-by-metadata` | 내부 | — | `server/routes/internal.py:vector_get_by_metadata` |
| POST | `/api/internal/vector/list-collections` | 내부 | — | `server/routes/internal.py:vector_list_collections` |
| POST | `/api/internal/vector/query` | 내부 | 내부 서비스를 위한 벡터 검색을 실행합니다. | `server/routes/internal.py:vector_query` |
| POST | `/api/internal/vector/update-metadata` | 내부 | — | `server/routes/internal.py:vector_update_metadata` |
| POST | `/api/internal/vector/upsert` | 내부 | — | `server/routes/internal.py:vector_upsert` |
| POST | `/api/internal/workspace/grant` | 내부 | 루트 소유 기록기를 통해 사람이 시작한 워크스페이스 권한을 적용합니다. | `server/routes/internal.py:internal_workspace_grant` |
| GET | `/api/messages/{message_id}` | 세션 필수(local_trust 모드이거나 localhost 신뢰가 활성화된 경우 생략 가능) | ID로 저장된 메시지의 전체 JSON을 반환합니다. | `server/routes/internal.py:get_message` |

## `server/routes/logs_routes.py`

| GET | `/api/system/logs` | 세션 필수(local_trust 모드이거나 localhost 신뢰가 활성화된 경우 생략 가능) | 사용 가능한 로그 파일 목록을 반환합니다. | `server/routes/logs_routes.py:list_logs` |
| GET | `/api/system/logs/file/read` | 세션 필수(local_trust 모드이거나 localhost 신뢰가 활성화된 경우 생략 가능) | 목록 엔드포인트에서 받은 기본 파일명이나 상대 경로로 로그 파일을 읽습니다. | `server/routes/logs_routes.py:read_log_by_ref` |
| GET | `/api/system/logs/stream` | 세션 필수(local_trust 모드이거나 localhost 신뢰가 활성화된 경우 생략 가능) | 실시간 로그 스트리밍을 위한 SSE 엔드포인트(tail -f 방식)입니다. | `server/routes/logs_routes.py:stream_logs` |
| GET | `/api/system/logs/{filename}` | 세션 필수(local_trust 모드이거나 localhost 신뢰가 활성화된 경우 생략 가능) | 페이지네이션을 적용해 로그 파일 내용을 읽습니다. | `server/routes/logs_routes.py:read_log` |

## `server/routes/memory_routes.py`

| DELETE | `/api/animas/{name}/conversation` | 세션 필수(local_trust 모드 또는 localhost 신뢰가 활성화된 경우 생략 가능) | 새로 시작할 수 있도록 대화 기록을 지웁니다. | `server/routes/memory_routes.py:clear_conversation` |
| GET | `/api/animas/{name}/conversation` | 세션 필수(local_trust 모드 또는 localhost 신뢰가 활성화된 경우 생략 가능) | 현재 대화 상태를 조회합니다. | `server/routes/memory_routes.py:get_conversation` |
| POST | `/api/animas/{name}/conversation/compress` | 세션 필수(local_trust 모드 또는 localhost 신뢰가 활성화된 경우 생략 가능) | 대화 압축을 수동으로 실행합니다. | `server/routes/memory_routes.py:compress_conversation` |
| GET | `/api/animas/{name}/episodes` | 세션 필수(local_trust 모드 또는 localhost 신뢰가 활성화된 경우 생략 가능) | — | `server/routes/memory_routes.py:list_episodes` |
| GET | `/api/animas/{name}/episodes/calendar` | 세션 필수(local_trust 모드 또는 localhost 신뢰가 활성화된 경우 생략 가능) | 한 달의 각 날짜에 에피소드가 있는지 간략하게 반환합니다. | `server/routes/memory_routes.py:episode_calendar` |
| GET | `/api/animas/{name}/episodes/{date}` | 세션 필수(local_trust 모드 또는 localhost 신뢰가 활성화된 경우 생략 가능) | — | `server/routes/memory_routes.py:get_episode` |
| GET | `/api/animas/{name}/knowledge` | 세션 필수(local_trust 모드 또는 localhost 신뢰가 활성화된 경우 생략 가능) | — | `server/routes/memory_routes.py:list_knowledge` |
| GET | `/api/animas/{name}/knowledge/{topic}` | 세션 필수(local_trust 모드 또는 localhost 신뢰가 활성화된 경우 생략 가능) | — | `server/routes/memory_routes.py:get_knowledge` |
| GET | `/api/animas/{name}/memory/connections` | 세션 필수(local_trust 모드 또는 localhost 신뢰가 활성화된 경우 생략 가능) | 활성 사실에서 추출한 엔터티 연결을 반환합니다. | `server/routes/memory_routes.py:memory_connections` |
| GET | `/api/animas/{name}/memory/connections/entity` | 세션 필수(local_trust 모드 또는 localhost 신뢰가 활성화된 경우 생략 가능) | 단일 엔터티에 대한 사실과 관련 메모리 파일을 반환합니다. | `server/routes/memory_routes.py:memory_connection_entity` |
| GET | `/api/animas/{name}/memory/graph` | 세션 필수(local_trust 모드 또는 localhost 신뢰가 활성화된 경우 생략 가능) | UI 표시에 사용할 명시적 링크 메모리 그래프를 반환합니다. | `server/routes/memory_routes.py:memory_graph` |
| GET | `/api/animas/{name}/memory/stats` | 세션 필수(local_trust 모드 또는 localhost 신뢰가 활성화된 경우 생략 가능) | anima의 메모리 저장소 통계를 반환합니다. | `server/routes/memory_routes.py:memory_stats` |
| GET | `/api/animas/{name}/procedures` | 세션 필수(local_trust 모드 또는 localhost 신뢰가 활성화된 경우 생략 가능) | — | `server/routes/memory_routes.py:list_procedures` |
| GET | `/api/animas/{name}/procedures/{proc}` | 세션 필수(local_trust 모드 또는 localhost 신뢰가 활성화된 경우 생략 가능) | — | `server/routes/memory_routes.py:get_procedure` |

## `server/routes/phone.py`

| GET | `/api/webhooks/twilio/audio/{token}.wav` | 불필요(제외 목록) | 추측할 수 없는 토큰으로 만료되지 않은 합성 WAV를 제공합니다. | `server/routes/phone.py:audio` |
| POST | `/api/webhooks/twilio/pin` | 불필요(제외 목록) | PIN을 확인하고 전화 스트림용 일회성 자격 증명을 발급합니다. | `server/routes/phone.py:pin` |
| POST | `/api/webhooks/twilio/status` | 불필요(제외 목록) | Twilio 통화 상태 콜백을 처리하고 종료된 세션을 해제합니다. | `server/routes/phone.py:status` |
| POST | `/api/webhooks/twilio/voice` | 불필요(제외 목록) | 수신 전화, 알림 재생 및 알림 DTMF 응답을 처리합니다. | `server/routes/phone.py:voice` |

## `server/routes/phone_stream.py`

| WS | `/api/webhooks/twilio/stream` | 불필요(제외 목록) | — | `server/routes/phone_stream.py:phone_media_stream` |

## `server/routes/sessions.py`

| GET | `/api/animas/{name}/conversation/bg` | 세션 필수(local_trust 모드 또는 localhost 신뢰가 활성화된 경우 생략 가능) | 접힌 피드 그룹 하나의 백그라운드 활동을 불러옵니다. | `server/routes/sessions.py:get_conversation_background` |
| POST | `/api/animas/{name}/conversation/calls` | 세션 필수(local_trust 모드 또는 localhost 신뢰가 활성화된 경우 생략 가능) | 완료된 음성 통화를 Anima의 대화 기록에 저장합니다. | `server/routes/sessions.py:record_voice_call` |
| GET | `/api/animas/{name}/conversation/feed` | 세션 필수(local_trust 모드 또는 localhost 신뢰가 활성화된 경우 생략 가능) | 백그라운드 활동 그룹을 지연 로드하는 사용자 중심 채팅 기록을 가져옵니다. | `server/routes/sessions.py:get_conversation_feed` |
| GET | `/api/animas/{name}/conversation/history` | 세션 필수(local_trust 모드 또는 localhost 신뢰가 활성화된 경우 생략 가능) | 활동 로그에서 대화 기록을 가져옵니다. | `server/routes/sessions.py:get_conversation_history` |
| GET | `/api/animas/{name}/conversation/tool-calls` | 세션 필수(local_trust 모드 또는 localhost 신뢰가 활성화된 경우 생략 가능) | assistant 메시지 하나의 전체 도구 호출 세부 정보를 잘라서 가져옵니다. | `server/routes/sessions.py:get_conversation_tool_calls` |
| POST | `/api/animas/{name}/notifications/reply` | 세션 필수(local_trust 모드 또는 localhost 신뢰가 활성화된 경우 생략 가능) | human_notify 항목에 대한 답변을 해당 항목을 보낸 Anima로 전달합니다. | `server/routes/sessions.py:reply_to_notification` |
| GET | `/api/animas/{name}/sessions` | 세션 필수(local_trust 모드 또는 localhost 신뢰가 활성화된 경우 생략 가능) | 사용 가능한 모든 세션(활성 대화, 보관 기록, 에피소드)을 나열합니다. | `server/routes/sessions.py:list_sessions` |
| GET | `/api/animas/{name}/sessions/{session_id}` | 세션 필수(local_trust 모드 또는 localhost 신뢰가 활성화된 경우 생략 가능) | 보관된 세션의 세부 정보를 가져옵니다. | `server/routes/sessions.py:get_session_detail` |
| GET | `/api/animas/{name}/transcripts/{date}` | 세션 필수(local_trust 모드 또는 localhost 신뢰가 활성화된 경우 생략 가능) | 특정 날짜의 전체 대화 기록을 가져옵니다. | `server/routes/sessions.py:get_transcript` |
| GET | `/api/notifications/awaiting` | 세션 필수(local_trust 모드 또는 localhost 신뢰가 활성화된 경우 생략 가능) | Anima별로 최근 24시간 동안 접수된 call_human 알림 중 아직 답변을 기다리는 항목을 나열합니다. | `server/routes/sessions.py:list_awaiting_notifications` |

## `server/routes/setup.py`

| POST | `/api/setup/codex/device-login` | 불필요(제외 목록) | Codex 로그인에 사용할 브라우저 기반 기기 인증 안내를 반환합니다. | `server/routes/setup.py:start_codex_device_login` |
| POST | `/api/setup/complete` | 불필요(제외 목록) | 설정을 완료합니다. 구성을 저장하고 anima를 생성한 뒤 완료 상태로 표시합니다. | `server/routes/setup.py:complete_setup` |
| GET | `/api/setup/detect-locale` | 불필요(제외 목록) | Accept-Language 헤더에서 로캘을 감지합니다. | `server/routes/setup.py:detect_locale` |
| GET | `/api/setup/environment` | 불필요(제외 목록) | 설정 마법사에 필요한 환경 정보를 반환합니다. | `server/routes/setup.py:get_environment` |
| POST | `/api/setup/validate-key` | 불필요(제외 목록) | 소규모 테스트 요청을 보내 API 키를 확인합니다. | `server/routes/setup.py:validate_key` |

## `server/routes/skills.py`

| GET | `/api/animas/{name}/skills` | 세션 필수(local_trust 모드 또는 localhost 신뢰가 활성화된 경우 생략 가능) | anima에서 표시되는 스킬을 나열하고 스레드별 활성 항목을 표시합니다. | `server/routes/skills.py:list_skills` |
| GET | `/api/animas/{name}/skills/active` | 세션 필수(local_trust 모드 또는 localhost 신뢰가 활성화된 경우 생략 가능) | 채팅 스레드에 현재 설정된 활성 스킬을 반환합니다. | `server/routes/skills.py:get_active_skills` |
| PUT | `/api/animas/{name}/skills/active` | 세션 필수(local_trust 모드 또는 localhost 신뢰가 활성화된 경우 생략 가능) | 채팅 스레드의 활성 스킬을 교체합니다. | `server/routes/skills.py:update_active_skills` |
| POST | `/api/animas/{name}/skills/trust` | 세션 필수(local_trust 모드 또는 localhost 신뢰가 활성화된 경우 생략 가능) | 안전한 스킬을 신뢰할 수 있는 운영 지침으로 승격합니다. | `server/routes/skills.py:trust_skill` |

## `server/routes/system.py`

| GET | `/api/activity/group` | 세션 필수(local_trust 모드 또는 localhost 신뢰가 활성화된 경우 생략 가능) | 안정적인 ID로 트리거 기반 활동 그룹 하나를 완전한 형태로 반환합니다. | `server/routes/system.py:get_activity_group` |
| GET | `/api/activity/recent` | 세션 필수(local_trust 모드 또는 localhost 신뢰가 활성화된 경우 생략 가능) | 통합 ActivityLogger에서 최근 활동 이벤트를 반환합니다. | `server/routes/system.py:get_recent_activity` |
| GET | `/api/activity/running-tasks` | 세션 필수(local_trust 모드 또는 localhost 신뢰가 활성화된 경우 생략 가능) | 활성 TaskExec 작업자와 기타 사용 중인 레인을 Anima별로 그룹화해 반환합니다. | `server/routes/system.py:get_running_activity_tasks` |
| GET | `/api/settings/activity-level` | 세션 필수(local_trust 모드 또는 localhost 신뢰가 활성화된 경우 생략 가능) | 현재 전역 활동 수준과 일정을 반환합니다. | `server/routes/system.py:get_activity_level` |
| PUT | `/api/settings/activity-level` | 세션 필수(local_trust 모드 또는 localhost 신뢰가 활성화된 경우 생략 가능) | 전역 활동 수준을 업데이트하고 모든 하트비트를 다시 예약합니다. | `server/routes/system.py:set_activity_level` |
| PUT | `/api/settings/activity-schedule` | 세션 필수(local_trust 모드 또는 localhost 신뢰가 활성화된 경우 생략 가능) | 시간 기반 활동 일정(야간 모드)을 업데이트합니다. | `server/routes/system.py:set_activity_schedule` |
| POST | `/api/settings/display-mode` | 세션 필수(local_trust 모드 또는 localhost 신뢰가 활성화된 경우 생략 가능) | 표시 모드를 업데이트하고 config.image_gen.image_style과 동기화합니다. | `server/routes/system.py:set_display_mode` |
| GET | `/api/shared/users` | 세션 필수(local_trust 모드 또는 localhost 신뢰가 활성화된 경우 생략 가능) | shared/users/.에서 등록된 사용자 이름을 나열합니다. | `server/routes/system.py:list_shared_users` |
| GET | `/api/system/connections` | 세션 필수(local_trust 모드 또는 localhost 신뢰가 활성화된 경우 생략 가능) | WebSocket 및 프로세스 연결 정보를 반환합니다. | `server/routes/system.py:system_connections` |
| GET | `/api/system/frontend-logs` | 세션 필수(local_trust 모드 또는 localhost 신뢰가 활성화된 경우 생략 가능) | 선택적 필터를 적용해 JSONL 파일에서 프런트엔드 로그를 읽습니다. | `server/routes/system.py:view_frontend_logs` |
| POST | `/api/system/frontend-logs` | 세션 필수(local_trust 모드 또는 localhost 신뢰가 활성화된 경우 생략 가능) | 프런트엔드 로그 항목을 일괄 수신해 일별 JSONL 파일에 기록합니다. | `server/routes/system.py:receive_frontend_logs` |
| GET | `/api/system/health` | 불필요(제외 목록) | 간단한 상태 확인 엔드포인트입니다. | `server/routes/system.py:health_check` |
| POST | `/api/system/hot-reload/credentials` | 세션 필수(local_trust 모드 또는 localhost 신뢰가 활성화된 경우 생략 가능) | 인증 정보와 이에 종속된 연결을 핫 리로드합니다. | `server/routes/system.py:hot_reload_credentials` |
| POST | `/api/system/hot-reload/slack` | 세션 필수(local_trust 모드 또는 localhost 신뢰가 활성화된 경우 생략 가능) | Slack Socket Mode 연결만 핫 리로드합니다. | `server/routes/system.py:hot_reload_slack` |
| GET | `/api/system/log-level` | 세션 필수(local_trust 모드 또는 localhost 신뢰가 활성화된 경우 생략 가능) | 현재 루트 로그 수준을 반환합니다. | `server/routes/system.py:get_log_level` |
| POST | `/api/system/log-level` | 세션 필수(local_trust 모드 또는 localhost 신뢰가 활성화된 경우 생략 가능) | 실행 중에 로그 수준을 변경합니다(재시작 불필요). | `server/routes/system.py:set_log_level` |
| POST | `/api/system/reload` | 세션 필수(local_trust 모드 또는 localhost 신뢰가 활성화된 경우 생략 가능) | 전체 동기화: 새 anima를 추가하고, 기존 항목을 새로 고치며, 삭제된 항목을 제거합니다. | `server/routes/system.py:reload_animas` |
| POST | `/api/system/rewrite-runtime-refs` | 세션 필수(local_trust 모드 또는 localhost 신뢰가 활성화된 경우 생략 가능) | REWRITE_REFS가 디스크 상태를 업데이트한 후 실시간 캐시를 동기화합니다. | `server/routes/system.py:rewrite_runtime_refs` |
| GET | `/api/system/scheduler` | 세션 필수(local_trust 모드 또는 localhost 신뢰가 활성화된 경우 생략 가능) | 스케줄러 상태와 작업 정보를 반환합니다. | `server/routes/system.py:system_scheduler` |
| GET | `/api/system/status` | 세션 필수(local_trust 모드 또는 localhost 신뢰가 활성화된 경우 생략 가능) | — | `server/routes/system.py:system_status` |
| GET | `/api/system/token-budget` | 세션 필수(local_trust 모드 또는 localhost 신뢰가 활성화된 경우 생략 가능) | 각 Anima의 이번 달 토큰 예산 상태를 반환합니다. | `server/routes/system.py:get_token_budget` |

## `server/routes/taskboard.py`

| GET | `/api/task-board` | 세션 필수(local_trust 모드 또는 localhost 신뢰가 활성화된 경우 생략 가능) | 정식 TaskStore에서 바로 읽은 통합 TaskBoard 뷰를 반환합니다. | `server/routes/taskboard.py:list_task_board` |
| GET | `/api/task-board/summary` | 세션 필수(local_trust 모드 또는 localhost 신뢰가 활성화된 경우 생략 가능) | 대시보드에서 사용할 TaskBoard 요약 수치를 반환합니다. | `server/routes/taskboard.py:get_task_board_summary` |
| POST | `/api/task-board/{anima_name}/{task_id}/cancel` | 세션 필수(local_trust 모드 또는 localhost 신뢰가 활성화된 경우 생략 가능) | 미해결 리스를 무시하고 작업을 명시적으로 취소합니다(사람이 수행하는 작업). | `server/routes/taskboard.py:cancel_task` |

## `server/routes/usage_routes.py`

| GET | `/api/usage` | 세션 필수(local_trust 모드 또는 localhost 신뢰가 활성화된 경우 생략 가능) | Claude + OpenAI + nanoGPT의 통합 사용량 데이터를 반환합니다. | `server/routes/usage_routes.py:get_usage` |
| POST | `/api/usage/claude/relogin` | 세션 필수(local_trust 모드 또는 localhost 신뢰가 활성화된 경우 생략 가능) | — | `server/routes/usage_routes.py:relogin_claude` |
| POST | `/api/usage/openai/relogin` | 세션 필수(local_trust 모드 또는 localhost 신뢰가 활성화된 경우 생략 가능) | — | `server/routes/usage_routes.py:relogin_openai` |

## `server/routes/users.py`

| GET | `/api/users` | 세션 필수(local_trust 모드 또는 localhost 신뢰가 활성화된 경우 생략 가능) | 모든 사용자를 나열합니다(비밀번호 해시 제외). | `server/routes/users.py:list_users` |
| POST | `/api/users` | 세션 필수(local_trust 모드 또는 localhost 신뢰가 활성화된 경우 생략 가능) | 새 사용자를 추가합니다(소유자만 가능). | `server/routes/users.py:add_user` |
| PUT | `/api/users/me/password` | 세션 필수(local_trust 모드 또는 localhost 신뢰가 활성화된 경우 생략 가능) | 현재 사용자의 비밀번호를 변경하거나 최초 설정합니다. | `server/routes/users.py:change_password` |
| DELETE | `/api/users/{username}` | 세션 필수(local_trust 모드 또는 localhost 신뢰가 활성화된 경우 생략 가능) | 사용자를 삭제합니다(소유자만 가능하며 본인은 삭제할 수 없음). | `server/routes/users.py:delete_user` |

## `server/routes/voice.py`

| WS | `/ws/voice/{name}` | 세션 필수(local_trust 모드 또는 localhost 신뢰가 활성화된 경우 생략 가능) | 특정 Anima의 음성 대화 WebSocket입니다. | `server/routes/voice.py:voice_websocket` |

## `server/routes/webhooks.py`

| POST | `/api/webhooks/chatwork` | 불필요(제외 목록) | Chatwork 웹훅 알림을 처리합니다. | `server/routes/webhooks.py:chatwork_webhook` |
| POST | `/api/webhooks/github` | 불필요(제외 목록) | GitHub 웹훅 이벤트를 인증하고 대기열에 추가합니다. | `server/routes/webhooks.py:github_webhook` |
| POST | `/api/webhooks/slack/events` | 불필요(제외 목록) | Slack 이벤트 구독을 처리합니다. | `server/routes/webhooks.py:slack_events` |
| POST | `/api/webhooks/zoom` | 불필요(제외 목록) | Zoom RTMS 웹훅 알림을 처리합니다. | `server/routes/webhooks.py:zoom_webhook` |
| GET | `/api/webhooks/zoom/oauth-callback` | 불필요(제외 목록) | Zoom OAuth 리디렉션 랜딩 페이지입니다. | `server/routes/webhooks.py:zoom_oauth_callback` |

## `server/routes/websocket_route.py`

| WS | `/ws` | 세션 필수(local_trust 모드 또는 localhost 신뢰가 활성화된 경우 생략 가능) | — | `server/routes/websocket_route.py:websocket_endpoint` |
