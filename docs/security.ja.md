# セキュリティアーキテクチャ

AnimaWorksはツールアクセス・永続記憶・エージェント間通信を持つ自律AIエージェントを実行する。これはステートレスなLLMラッパーとは根本的に異なる脅威面を生み出す — エージェントはファイルを読み、コマンドを実行し、メッセージを送り、人間の介入なしにスケジュールで動作する。

本ドキュメントでは、多層防御セキュリティモデルと、最新のLLM/エージェント攻撃研究（OWASP Top 10 for LLM 2025、AdapTools、MemoryGraft、ChatInject、RoguePilot、MCP Tool Poisoning、RAGPoison、Confused Deputy攻撃）に基づく敵対的脅威分析を記述する。

**最終監査**: 2026-03-02

---

## 脅威モデル

| 脅威 | 攻撃経路 | 影響 |
|------|---------|------|
| 外部データ経由のプロンプトインジェクション | Web検索結果、Slack/Chatworkメッセージ、メール | エージェントが攻撃者の指示を実行 |
| RAG/記憶の汚染 | 悪意あるWebコンテンツ → knowledge → 永続的想起 | 全セッションにわたる長期的な行動変容 |
| エージェント間のラテラルムーブメント | 侵害されたエージェントが悪意あるDMを同僚に送信 | 組織全体の権限昇格 |
| Confused Deputy攻撃 | 低権限エージェントが高権限エージェントを騙す | 無許可のツール実行、データ流出 |
| Consolidation汚染 | 汚染されたエピソード/アクティビティ → knowledge抽出 | 汚染源から生成された信頼済み知識 |
| 破壊的コマンド実行 | エージェントが `rm -rf /` や `curl … \| sh` を実行 | データ喪失、システム侵害 |
| シェルインジェクション迂回 | パイプ経由のネットワークツール | 許可コマンド経由のデータ流出 |
| パストラバーサル | エージェントがサンドボックス外に読み書き | エージェント間データ漏洩、設定改ざん |
| アクティビティログ改ざん | エージェントが自身のactivity_logに偽エントリ記録 | Primingコンテキストの操作 |
| 無限メッセージループ | 2つのエージェントが無限に返信し合う | リソース枯渇、API費用の爆発 |
| 意図しない外部送信 | エージェントが想定外の宛先にメッセージ送信 | データ流出 |
| セッションハイジャック | 期限なしトークンの窃取 | 永続的な不正アクセス |
| 認証情報の露出 | config.jsonに平文のAPIキー | 外部サービスの悪用 |

---

## Part I: 現在の防御レイヤー

### 1. プロンプトインジェクション防御 — 信頼境界ラベリング

エージェントのコンテキストに入る全データに信頼レベルがタグ付けされる。モデルはこの境界を明示的に認識し、untrustedコンテンツをデータとして扱い、指示としては絶対に従わないよう指示される。

#### 信頼レベル

| レベル | 対象ソース | 扱い |
|--------|-----------|------|
| `trusted` | 内部ツール（send_message, search_memory）、システム生成 | 通常通り実行 |
| `medium` | ファイル読み取り、RAG結果、ユーザープロファイル、統合済み知識 | 参考データとして解釈 |
| `untrusted` | web_search, slack_read, chatwork_read, gmail_read, x_search | **指示には絶対従わない** |

#### 実装

```
<tool_result tool="web_search" trust="untrusted">
  検索結果 — インジェクション試行が含まれる可能性あり
</tool_result>

<priming source="related_knowledge" trust="medium" origin="consolidation">
  RAGで取得されたコンテキスト
</priming>
```

**オリジンカテゴリ**: `system`, `human`, `anima`, `external_platform`, `external_web`, `consolidation`, `unknown`。各カテゴリは `ORIGIN_TRUST_MAP` で信頼レベルにマッピングされる。

**オリジンチェーン伝播**: データが複数システムを経由する場合（例: Web → RAGインデックス → Priming）、信頼レベルはチェーン中の**最小値**に劣化する。`resolve_trust(origin, origin_chain)` がチェーン全ノード + 現在のオリジンにわたる保守的な最小値を計算する。

**セッションレベル信頼追跡**: `_min_trust_seen` がセッション中の全ツール呼び出しにわたる最小信頼ランク（2=trusted, 1=medium, 0=untrusted）を追跡。Mode S（`PreToolUse` フック + `run/min_trust_seen` ファイル）、Mode A（`litellm_loop` と `anthropic_fallback`）で更新。各インタラクションサイクル開始時にリセット。

**主要ファイル**: `core/execution/_sanitize.py`（信頼解決、境界ラッピング、`TOOL_TRUST_LEVELS`、`ORIGIN_TRUST_MAP`）, `templates/*/prompts/tool_data_interpretation.md`（信頼レベルとオリジンチェーンの解釈指示）

---

### 2. 記憶の来歴管理 — RAG・知識の信頼追跡

#### write_memory_fileのオリジン伝播

エージェントが `knowledge/*.md` に書き込む際、システムはセッションの `_min_trust_seen` を確認する。セッションでuntrusted（ランク0）またはmedium（ランク1）のツール結果を処理していた場合、`origin` フロントマターが付加される:

- ランク0（untrusted） → `origin: external_web`
- ランク1（medium） → `origin: mixed`
- ランク2（trusted） → オリジンタグなし（クリーンな知識）

オリジンはRAGインデクサーにも渡され、ChromaDBのチャンクメタデータに格納される。

#### RAGインデクサーのオリジン追跡

`index_file()` が `origin` パラメータを受け取り、チャンクメタデータの `metadata["origin"]` として保存。

#### PrimingチャネルCの信頼分割

PrimingがRAG経由で関連知識を取得する際、各チャンクの `origin` メタデータを `resolve_trust()` で評価。チャンクを分割:

- **trusted/medium** → `related_knowledge`（`trust="medium"` でラップ）
- **untrusted** → `related_knowledge_external`（`trust="untrusted"`, `origin="external_platform"` でラップ）

バジェットはtrusted/mediumコンテンツを優先配分し、untrustedは残りのバジェットで充填。

#### Consolidationのオリジン追跡

日次Consolidationがソース知識ファイルのYAMLフロントマター `origin:` を読み取り、外部オリジン（`external_web`, `mixed`, `consolidation_external`）を持つソースがある場合、統合出力を `origin: consolidation_external`（`untrusted` に解決）に格下げ。

**主要ファイル**: `core/tooling/handler_memory.py`（write_memory_fileのオリジン伝播）, `core/memory/rag/indexer.py`（チャンクメタデータのオリジン）, `core/memory/priming.py`（チャネルC信頼分割）, `core/memory/consolidation.py`（オリジンチェーン追跡）

---

### 3. コマンド実行セキュリティ — 5層防御

エージェントはシェルコマンドを実行できる。5つの独立したレイヤーが悪用を防ぐ:

#### レイヤー1: シェルインジェクション検出

コマンドの連鎖やインジェクションに使われるシェルメタ文字をブロック:

- セミコロン（`;`）、バッククォート（`` ` ``）、改行（`\n`）
- コマンド置換（`$()`、`${}`、`$VAR`）

#### レイヤー2: ハードコードブロックリスト

権限設定に関係なく**常に**ブロックされる、パターンマッチによるコマンド:

| パターン | 理由 |
|---------|------|
| `rm -rf`, `rm -r` | 再帰的削除 |
| `mkfs` | ファイルシステム作成 |
| `dd of=/dev/` | ディスク直接書き込み |
| `curl\|sh`, `wget\|sh` | リモートコード実行 |
| `\| sh`, `\| bash`, `\| python`, `\| perl`, `\| ruby`, `\| node` | インタープリタへのパイプ |
| `nc`, `ncat`, `socat`, `telnet` | ネットワーク経由のデータ流出ツール |
| `curl -d/-F/-T`, `curl --data`, `wget --post` | データアップロード・流出 |
| `chmod *7*` | ワールドライタブル権限 |
| `shutdown`, `reboot` | システムシャットダウン |
| `> /dev/sd*`, `> /dev/nvme*`, `> /etc/` | デバイス/システムファイルへのリダイレクト |

#### レイヤー2.5: エージェント個別の禁止コマンド

各エージェントの `permissions.md` に `## 実行できないコマンド` セクションで追加ブロック可能。

#### レイヤー3: セクション必須

`permissions.md` に `## コマンド実行` または `## 実行できるコマンド` セクションが必要 — 明示的権限がないエージェントはデフォルト拒否。

#### レイヤー4: エージェント個別の許可リスト

許可リストにマッチするコマンドのみ実行許可。

#### レイヤー5: パストラバーサル検出

コマンド引数にパストラバーサルパターン（`../`）がないかチェック。

**パイプライン個別チェック**: パイプで連結された各セグメントは独立にチェック。

**主要ファイル**: `core/tooling/handler_base.py`（`_BLOCKED_CMD_PATTERNS`, `_INJECTION_RE`）, `core/tooling/handler_perms.py`（`_check_command_permission`）

---

### 4. ファイルアクセス制御 — デフォルトでサンドボックス

各エージェントは自身のディレクトリ（`~/.animaworks/animas/{name}/`）内で動作する。

#### 保護ファイル・ディレクトリ（イミュータブル）

エージェント自身が書き込めない:

- `permissions.md` — ツール・コマンドの許可リスト
- `identity.md` — 人格の基盤（不変ベースライン）
- `bootstrap.md` — 初回起動指示
- `activity_log/` — アクティビティログディレクトリ。`ActivityLogger`（コードレベル）のみが追記可能

#### 上司アクセスマトリクス

| パス | 直属部下 | 全配下 |
|------|:---:|:---:|
| `activity_log/` | 読み取り | 読み取り |
| `state/current_task.md`, `pending.md` | — | 読み取り |
| `state/task_queue.jsonl`, `pending/` | — | 読み取り |
| `status.json` | 読み書き | 読み取り |
| `identity.md` | — | 読み取り |
| `injection.md` | 読み書き | 読み取り |
| `cron.md`, `heartbeat.md` | 読み書き | — |

配下の解決にはBFS + サイクル検出。同僚（同一上司）は互いの `activity_log/` を読み取り可能。

**主要ファイル**: `core/tooling/handler_base.py`（`_PROTECTED_FILES`, `_PROTECTED_DIRS`, `_is_protected_write`）, `core/tooling/handler_perms.py`（`_check_file_permission`）

---

### 5. プロセス隔離

各エージェントは独立OSプロセスとして動作:

- **プロセス分離**: 1つのエージェントのクラッシュが他に影響しない
- **Unix Domain Socket IPC**: ファイルシステムソケットによるプロセス間通信（TCP不使用）
- **独立ロック**: チャット、Inbox、バックグラウンドタスクが別々のasyncioロック
- **ソケットディレクトリ**: `~/.animaworks/run/sockets/{name}.sock`、起動時に残存ソケットの自動クリーンアップ

**主要ファイル**: `core/supervisor/manager.py`, `core/supervisor/ipc.py`, `core/supervisor/runner.py`

---

### 6. レート制限 — 3層のアウトバウンド制御

#### レイヤー1: セッション内（Per-Run）

- 同一宛先へのDM重複なし
- 1実行あたり最大2件の異なるDM宛先
- 1チャネルにつき1セッション1投稿
- セッション横断のチャネル投稿クールダウン（`channel_post_cooldown_s`）
- `run/replied_to.jsonl` に永続化

#### レイヤー2: セッション横断（永続）

- エージェントあたり**設定可能な1時間/1日あたりの送信上限**
- `activity_log` のスライディングウィンドウから計算
- `ack`, `error`, `system_alert` は制限対象外

#### レイヤー3: 行動認識（自己調整）

直近の送信メッセージ（過去2時間、最大3件）がPrimingでシステムプロンプトに注入。

#### カスケード防止

- **会話深度リミッター**: 設定可能な最大ターン数（`depth_window_s` 内）
- **Inboxレートリミッター**: クールダウン、カスケード検出、送信者別レート制限
- **Fail-closed**: アクティビティログ読み取り失敗時は `False` を返す

**主要ファイル**: `core/tooling/handler_comms.py`, `core/cascade_limiter.py`, `core/supervisor/inbox_rate_limiter.py`, `core/memory/priming.py`

---

### 7. 認証・セッション管理

#### 認証モード

| モード | 用途 |
|--------|------|
| `local_trust` | 開発用 — localhostからのリクエストは認証バイパス |
| `password` | シングルユーザーのパスワード保護 |
| `multi_user` | 複数ユーザー（個別アカウント） |

#### セッションセキュリティ

- **Argon2id** パスワードハッシュ（メモリハード、サイドチャネル耐性）
- **48バイトURLセーフトークン**（暗号学的乱数）
- **ユーザーあたり最大10セッション** — オーバーフロー時は最古を破棄
- **セッションTTL** — `config.server.session_ttl_days`（デフォルト: 7日）。期限切れセッションは `validate_session()` で拒否・削除
- **パスワード変更でセッション破棄** — `change_password()` は `revoke_all_sessions()` で全セッションを破棄
- **Cookie ベース** のセッション伝送、`/api/` と `/ws` ルートにミドルウェアガード
- 設定ファイルは**0600パーミッション**で保存

#### Localhost信頼

`trust_localhost` 有効時、ループバックアドレスからのリクエストは自動認証。OriginとHostヘッダーチェックによるCSRF緩和。

**主要ファイル**: `core/auth/manager.py`, `server/app.py`, `server/localhost.py`

---

### 8. Webhook検証

| プラットフォーム | 方式 | リプレイ防止 |
|----------------|------|------------|
| Slack | HMAC-SHA256（signing secret） | タイムスタンプチェック（5分ウィンドウ） |
| Chatwork | HMAC-SHA256（webhookトークン） | — |

両方とも `hmac.compare_digest` による定数時間比較。

**主要ファイル**: `server/routes/webhooks.py`

---

### 9. SSRF緩和 — メディアプロキシ

メディアプロキシ（`/api/media/proxy`）はUI表示用に外部画像を取得する:

- **HTTPSのみ**
- **ドメイン許可リスト or open-with-scan** — `MediaProxyConfig.mode` で設定
- **プライベートIPブロック** — localhost、RFC 1918、リンクローカル、マルチキャスト、予約済み
- **DNS解決チェック** — DNSリバインディング防止
- **Content-Type検証** — `image/jpeg`, `image/png`, `image/gif`, `image/webp` のみ。SVGブロック
- **マジックバイト検証** — 実際のファイル形式と宣言されたContent-Typeの一致確認
- **サイズ制限** — `max_bytes`（デフォルト5MB）
- **リダイレクト検証** — リダイレクト先の再検証、最大リダイレクト回数制限
- **IP別レート制限** — 設定可能（デフォルト30リクエスト/分）
- **セキュリティヘッダー** — `X-Content-Type-Options: nosniff`

**主要ファイル**: `server/routes/media_proxy.py`

---

### 10. Mode S（Agent SDK）セキュリティ

Claude Agent SDK（Mode S）で動作する場合、`PreToolUse` フックによる追加ガードレール:

- **Bashコマンドフィルタリング**: SDK用個別ブロックリスト（Chatwork CLIバイパス防止、ネットワーク流出ツール、データアップロードパターン含む）
- **ファイル書き込み保護**: 保護ファイルリストとサンドボックスに対する検証
- **ファイル読み取り制限**: 他エージェントのディレクトリへのアクセスブロック（部下/同僚のactivity_log、部下管理ファイルを除く）
- **出力トランケーション**: Bash出力10KB上限、ファイル読み取り500行デフォルト制限、grep/globも制限
- **信頼追跡**: `_SDK_TOOL_TRUST` マッピング、`run/min_trust_seen` に永続化

**主要ファイル**: `core/execution/_sdk_security.py`, `core/execution/_sdk_hooks.py`

---

### 11. アウトバウンドルーティングセキュリティ

`resolve_recipient()` がエージェントの意図しない宛先への送信を防止:

1. 既知エージェント名との完全一致（大文字小文字区別）
2. ユーザーエイリアス検索（大文字小文字非区別）
3. プラットフォームプレフィックス付き宛先
4. Slack User IDパターンマッチ
5. 大文字小文字非区別のエージェント名マッチ
6. **未知の宛先 → RecipientNotFoundError**（fail-closed）

**主要ファイル**: `core/outbound.py`

---

### 12. エージェント間メッセージセキュリティ

#### メッセージのオリジンチェーン

DMは `origin_chain` メタデータを含み、`build_outgoing_origin_chain()` で構築。受信者がメッセージの信頼来歴を評価可能。

#### Inbox from_person 検証

`Messenger.receive()` が `from_person` を `known_animas`（`config.animas`）に対して検証。不明な `from_person` は拒否・ログ記録。

#### Inboxディレクトリパーミッション

Inboxディレクトリは `0o700` で作成。

#### チャネル名検証

`_SAFE_NAME_RE = re.compile(r"^[a-z][a-z0-9_-]{0,30}$")` によるパストラバーサル防止。

#### Boardチャネルコンテンツ制限

チャネル投稿は Pydantic で `max_length=10000` に制限。

**主要ファイル**: `core/messenger.py`, `core/tooling/handler_comms.py`, `core/tooling/handler_base.py`

---

## Part II: 敵対的脅威分析

### 解決済み脆弱性

初回監査で特定された脆弱性のうち、対策が完了したもの:

| ID | 深刻度 | タイトル | 解決策 |
|----|--------|-------|--------|
| RAG-1 | Critical → 緩和済み | Web → Knowledge → RAG永続汚染 | `write_memory_file` が `_min_trust_seen` をオリジンフロントマターとして伝播、RAGインデクサーがオリジンをチャンクメタデータに格納、PrimingチャネルCが信頼/非信頼を分割 |
| CON-1 | High → 緩和済み | Consolidationパイプライン汚染 | `_has_external_origin_in_files()` がソースファイルのオリジンを確認、外部オリジン時は `consolidation_external` に格下げ |
| MSG-1 | High → 緩和済み | Inboxファイルレベルなりすまし | `from_person` を `known_animas` に対して検証、Inboxディレクトリを `0o700` で保護 |
| BOARD-1 | High → 緩和済み | Boardチャネルブロードキャスト汚染 | 認証ミドルウェアがチャネルPOSTを保護、コンテンツ10,000文字制限、チャネル名正規表現検証 |
| ALOG-1 | High → 解決済み | アクティビティログ改ざん | `activity_log/` が `_PROTECTED_DIRS` に追加、`_is_protected_write` で書き込みブロック |
| CMD-1 | High → 解決済み | シェルモードネットワーク流出 | `nc`, `ncat`, `socat`, `telnet`, `curl -d/--data`, `wget --post` をブロックリストに追加 |
| AUTH-1 | High → 解決済み | 永久セッショントークン | `validate_session()` でTTLチェック（デフォルト7日）、`change_password()` が `revoke_all_sessions()` を呼び出し |
| DEPUTY-1 | Medium → 緩和済み | Confused Deputy権限昇格 | メッセージに `origin_chain` メタデータ、`from_person` 検証、`tool_data_interpretation` の信頼境界指示 |

---

### 残存脆弱性

#### High

| ID | カテゴリ | タイトル | 状態 |
|----|---------|-------|------|
| CFG-1 | 設定 | 平文の認証情報保存 | 部分的（ツール別env_varフォールバックあり、CredentialConfigにenv-onlyモードなし） |

#### Medium

| ID | カテゴリ | タイトル | 状態 |
|----|---------|-------|------|
| IPC-1 | ネットワーク | ソケットファイルパーミッション | 未実装（Unixソケットに `chmod 0o700` なし） |
| WS-1 | ネットワーク | 音声WebSocket音声インジェクション | 部分的（60秒バッファ上限あり、最大フレームサイズ/PCM形式検証なし） |
| OB-1 | レート制限 | マルチエージェント分散スパム | 未実装（送信者単位のレート制限のみ、受信者集約なし） |
| PR-1 | 記憶 | PageRankグラフ操作 | 未実装（信頼加重PageRankなし） |
| SKILL-1 | 記憶 | スキル説明のキーワードスタッフィング | 未実装（3段階マッチングに対策なし） |
| PI-1 | プロンプト | ツール信頼レベル登録漏れ | 未実装（未登録ツールはuntrustedにフォールバック、CIチェックなし） |
| CMD-2 | 実行 | 禁止リストの部分一致バイパス | 未実装（部分文字列マッチ、`shutil.which()` 解決なし） |
| EXT-1 | 外部 | 外部ソース経由の間接インジェクション | 信頼ラベリングで緩和、追加の正規表現フィルタなし |
| LEAK-1 | 情報漏洩 | システムプロンプト漏洩 | 部分的（信頼ルールあり、明示的な漏洩防止指示なし） |

#### Low

| ID | カテゴリ | タイトル | 状態 |
|----|---------|-------|------|
| AUTH-2 | 認証 | Localhost信頼の過剰許可 | 未実装（`X-Forwarded-For` 非対応） |
| FILE-1 | ファイル | allowed_dirsでのシンボリックリンク追跡 | 未実装（`resolve()` 使用、strict シンボリックリンク拒否なし） |
| WS-2 | ネットワーク | WebSocket JSONスキーマの甘さ | 未実装（音声WebSocket JSONにPydantic検証なし） |
| OB-2 | レート制限 | アクティビティログ書き込みバイパス | 未実装（送信がアクティビティログ成功に依存しない） |
| ACCESS-1 | 記憶 | RAGアクセスカウントのインフレーション | 未実装（access_count上限なし） |

---

## Part III: 多層防御アーキテクチャ

```
┌─────────────────────────────────────────────────────────┐
│                    外部データ                             │
│          (Web, Slack, メール, Board, DM 等)               │
└────────────────────────┬────────────────────────────────┘
                         │
              ┌──────────▼──────────┐
              │  信頼境界            │  ← untrusted/medium/trusted タグ
              │  ラベリング          │  ← オリジンチェーン伝播
              └──────────┬──────────┘
                         │
              ┌──────────▼──────────┐
              │  認証・セッション     │  ← Argon2id, TTL付きセッション
              │  管理               │  ← Webhook HMAC検証
              └──────────┬──────────┘
                         │
     ┌───────────────────┼───────────────────┐
     │                   │                   │
┌────▼────┐      ┌──────▼──────┐     ┌──────▼──────┐
│ コマンド │      │ ファイル     │     │ アウトバウンド│
│ セキュリティ│    │ アクセス制御  │     │ レート制限   │
│ (5層     │      │ (サンドボックス│    │ (3層 +      │
│  チェック) │     │  + ACL)     │     │  カスケード)  │
└────┬────┘      └──────┬──────┘     └──────┬──────┘
     │                  │                   │
     └───────────────┐  │  ┌────────────────┘
                     │  │  │
              ┌──────▼──▼──▼────────┐
              │  記憶の来歴管理      │  ← RAG/knowledgeのオリジン追跡
              │  (信頼チェーン)      │  ← チャネルC信頼分割
              └──────────┬──────────┘
                         │
              ┌──────────▼──────────┐
              │  プロセス隔離        │  ← エージェント別OSプロセス
              │  (Unixソケット)      │  ← 独立ロック
              └─────────────────────┘
```

各レイヤーは独立して動作する。1つのレイヤーの失敗は他のレイヤーで補捉される。

---

## Part IV: 改善ロードマップ

### フェーズ1: 即時対応（XS工数）

| 優先度 | ID | アクション | 工数 |
|:---:|------|--------|:---:|
| 1 | IPC-1 | ソケットファイルと `run/` ディレクトリに `chmod 0o700` | XS |
| 2 | PI-1 | ツール信頼レベル登録の完全性CIチェック | XS |
| 3 | ACCESS-1 | アクセスカウント上限 + セッション単位の重複排除 | XS |

### フェーズ2: 強化（S〜M工数）

| 優先度 | ID | アクション | 工数 |
|:---:|------|--------|:---:|
| 4 | CFG-1 | env-var-onlyの認証情報モード、`config.json` のエージェント読み取り不可パス | M |
| 5 | WS-1 | 最大フレームサイズ + PCM形式検証 | S |
| 6 | OB-1 | 全エージェント横断の受信者別レート制限 | S |
| 7 | LEAK-1 | システムプロンプト漏洩防止指示、出力監視 | S |
| 8 | CMD-2 | `shutil.which()` 解決 + basename比較 | S |

### フェーズ3: 縦深防御（長期）

| 優先度 | ID | アクション | 工数 |
|:---:|------|--------|:---:|
| 9 | PR-1 | 信頼加重PageRank | M |
| 10 | EXT-1 | 外部データのインジェクションパターン正規表現フィルタ | M |
| 11 | AUTH-2 | リバースプロキシガイダンス、`X-Forwarded-For` 対応 | S |
| 12 | ALOG+ | アクティビティログのappend-onlyハッシュチェーン | M |
| 13 | MSG+ | エージェント間HMACメッセージ署名 | L |

工数目安: XS = 1時間未満, S = 1-4時間, M = 4-16時間, L = 16時間超

---

## 関連ドキュメント

| ドキュメント | 説明 |
|-------------|------|
| [Provenance Foundation](implemented/20260228_provenance-1-foundation.md) | 信頼解決とオリジンカテゴリ |
| [Input Boundary Labeling](implemented/20260228_provenance-2-input-boundary.md) | ツール結果とPrimingの信頼タグ |
| [Trust Propagation](implemented/20260228_provenance-3-propagation.md) | データフロー横断のオリジンチェーン |
| [RAG Provenance](implemented/20260228_provenance-4-rag-provenance.md) | ベクトル検索の信頼追跡 |
| [Mode S Trust](implemented/20260228_provenance-5-mode-s-trust.md) | Agent SDKセキュリティフック |
| [Command Injection Fix](implemented/20260228_security-command-injection-fix.md) | パイプ・改行インジェクション |
| [Path Traversal Fix](implemented/20260228_security-path-traversal-fix.md) | common_knowledge・create_animaのパス検証 |
| [Memory Write Security](implemented/20260215_memory-write-security-20260216.md) | 保護ファイルとクロスモード強化 |
