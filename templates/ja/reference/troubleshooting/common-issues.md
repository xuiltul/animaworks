# よくある問題と対処法

業務中に遭遇しやすい問題と、その対処手順をまとめたリファレンス。
各問題は「症状 → 原因 → 対処手順」の形式で記載されている。

困ったときは、まずこのドキュメントを読み、該当する項目の手順に従うこと。
ここで解決しない場合は `troubleshooting/escalation-flowchart.md` を参照して適切にエスカレーションすること。

---

## メッセージが届かない

### 症状

- 送信したはずのメッセージに返答がない
- 相手にメッセージが届いていないと言われた
- `send_message` を実行したが、相手が反応しない

### 原因

1. 宛先指定の誤り（Anima 正式名・ユーザーエイリアス・`slack:` / `chatwork:` プレフィックス等）や、解決順に合わない指定
2. サーバーが停止している
3. 相手の Anima が停止・無効化されている、または Inbox の Provider エラー待機中
4. 宛先解決、権限、外部チャネルの配送など実際の送信エラーが発生した
5. `intent` が未指定または不正。DM では `report` / `question` のみ。タスク委譲は `delegate_task` を使う（`send_message` に `intent="delegation"` を付けると非推奨メッセージが返る）
6. 同一 run で同じ宛先へ既に DM を送信済み（同一宛先への2通目のみ拒否。宛先数の上限はない）

### 対処手順

1. **送信先の名前・宛先形式を確認する**
   - `send_message` の `to` パラメータが意図した相手に解決されるか確認する
   - 実装（`core/messaging/outbound.py` `resolve_recipient`）の解決順は概ね次のとおり:
     1. 既知 Anima 名との**完全一致**（大文字小文字区別）→ 内部
     2. `config.json` `external_messaging.user_aliases` の**エイリアス**（大文字小文字無視）→ 外部（preferred_channel）
     3. `slack:USERID` / `chatwork:ROOMID` → 外部直接
     4. ベア Slack ユーザー ID（`U` + 英数字 8 文字以上）→ Slack 直接
     5. 既知 Anima 名の**大文字小文字無視一致** → 内部
     6. 上記以外 → 解決失敗
   - 内部 Anima へ確実に届けるには、`~/.animaworks/animas/<名前>/` または `reference/organization/structure.md` 上の**正式名**を使う
   - 確認方法:
     ```
     search_memory(query="組織", scope="common_knowledge")
     ```
     または `read_memory_file(path="reference/organization/structure.md")` で組織の全Anima名を確認
   - **注意**: チャット中に人間宛ての場合は `send_message` は使えない。直接テキストで返答すれば人間に届く。チャット以外（heartbeat等）で人間に連絡する場合は `call_human` を使用

2. **サーバーの稼働状況を確認する**
   - 自分が動作している時点でサーバーは稼働中のはず
   - それでも不安な場合は上司に「メッセージが届かない」旨を報告する

3. **相手の応答を待つ**
   - 相手はハートビート間隔（例: 30分ごと）で受信箱を確認する
   - 即座に返信がなくても、次のハートビートで処理される
   - 緊急の場合は上司に「至急連絡を取りたい」と報告し、手動起動を依頼する

4. **送信エラーが出た場合**
   - エラーメッセージを記録する
   - `state/current_state.md` に状況を記載する
   - 上司に報告する

### 具体例

```
# 名前を間違えていた場合
send_message(to="Aoi", content="...", intent="report")   # OK
send_message(to="aoi", content="...", intent="report")  # 名前が異なればエラーになる可能性あり

# DM は intent 必須（report / question のみ）。委譲は delegate_task
# 同一 run 内で同一宛先へ送れる DM は1通まで。宛先数の上限はない
send_message(
    to="aoi",
    content="了解しました。作業を開始します。",
    intent="report",           # 必須: report / question
    reply_to="msg-abc123",     # 任意: 元メッセージのID
    thread_id="thread-xyz789"  # 任意: スレッドID
)

# 確認・お礼・称賛だけのメッセージには返信しない。全体共有が必要なら post_channel（Board）を使用
```

---

## タスクを進められない

「条件が揃うまで待つ」「ブロック中」という状態は存在しない（`blocked` は廃止済み）。選べるのは進める・閉じる・相談するの3つだけ。

### 症状

- 作業を進めようとしたが、必要な情報や権限が不足している
- 他のAnimaの作業完了を待っている状態
- 外部サービスがエラーを返す

### 原因

1. 依存タスクが未完了
2. 権限不足（`permissions.json` で許可されていない操作。`permissions.md` のみの環境でも、初回 `load_permissions` で JSON が生成され MD は `.bak` に退避される）
3. 必要な情報が不足している
4. 外部サービスの障害

### 対処手順

1. **同じ操作を繰り返さない**
   - 何が足りないのかを具体的に特定する
   - 「誰の」「何の作業が」「いつまでに」必要かを整理する

2. **自力で解決できるか判断する**
   - 別のアプローチで回避できないか検討する
   - 記憶を検索して過去に同様の問題がなかったか確認する:
     ```
     search_memory(query="エラー内容やキーワード", scope="episodes")
     search_memory(query="回避", scope="knowledge")
     ```

3. **解決できない場合は依頼者に報告する**（`troubleshooting/escalation-flowchart.md` 参照）。報告には以下を含めること:
   - 何をしようとしたか
   - 何が足りないのか・何を待っているのか
   - 自分で試みた対処と推奨
   ```
   send_message(
       to="上司の名前",
       content="【進捗報告】\nタスク: XXXの実装\n事実: YYYのAPI権限が不足\n試行: permissions.jsonを確認したが該当設定なし\n推奨: API権限の追加をお願いします",
       intent="report"
   )
   ```

4. **タスクの扱いを決める**
   - 続ける可能性があるなら何もしなくてよい（タスクは `pending` のまま。宣言なしでセッションが終わっても自動で `pending` に戻る）
   - 不要になったら `update_task(status="cancelled", summary="理由")` にする

5. **待っている間に進められる作業がないか確認する**
   - 永続タスクキュー: ツールが使える場合は `list_tasks`、または `Bash: animaworks-tool task list` で確認する
   - `list_tasks(detail=true)` で未着手・依存関係・要対応理由を確認する。既存タスクの複製や手動での再投入はしない
   - 別のタスクに着手する

---

## 記憶が見つからない

### 症状

- 過去にやったことを思い出せない
- 手順書があるはずなのに見つからない
- 検索しても該当する結果が返ってこない

### 原因

1. 検索キーワードが適切でない
2. 検索スコープ（scope）が狭すぎる
3. まだ記憶として書き込まれていない（初めての作業）
4. ファイルパスを間違えている

### 対処手順

1. **スコープを広げて再検索する**
   - まず `all` スコープで広く検索する:
     ```
     search_memory(query="検索したいキーワード", scope="all")
     ```
   - 結果が多すぎる場合はスコープを絞る:
     ```
     search_memory(query="Slack設定", scope="procedures")    # 手順書に限定
     search_memory(query="Slack障害", scope="episodes")      # 過去の出来事に限定
     search_memory(query="Slack", scope="knowledge")         # 学んだ知識に限定
     ```

2. **キーワードを変えて再検索する**
   - 同義語や関連語で試す（例: 「送信」「メッセージ」「通知」「連絡」）
   - 英語キーワードでも試す（例: "slack", "message", "send"）
   - 部分一致を意識する（例: 「Chatwork」→「chatwork」「チャットワーク」）

3. **共有知識を検索する**
   - 個人の記憶にない場合、共有知識に存在する可能性がある:
     ```
     search_memory(query="検索キーワード", scope="common_knowledge")
     ```
   - 共有知識の目次を確認する:
     ```
     read_memory_file(path="common_knowledge/00_index.md")
     ```

4. **ディレクトリを直接確認する**
   - Mode S（Claude Agent SDK）などでは組み込み `Glob` で Anima ディレクトリ配下を一覧できる。Mode A 等では `read_memory_file` で既知のパスを開くか、`search_memory` で広く当たる
   - ファイル名が分かっていれば直接読む:
     ```
     read_memory_file(path="procedures/slack-setup.md")
     read_memory_file(path="knowledge/xxx-findings.md")
     ```

5. **記憶が存在しない場合**
   - 初めての作業である可能性がある
   - 共有知識（`common_knowledge/`）に関連ガイドがないか確認する
   - 上司や同僚に知見がないか問い合わせる
   - 作業完了後は MUST で記憶として記録する（次回のために）
   - 古い・重複した記憶は `archive_memory_file(path="...", reason="...")` で archive/ に退避できる（削除ではなく移動。`reason` は必須）

### 検索スコープ一覧

| scope | 検索対象 | 用途 |
|-------|---------|------|
| `knowledge` | 学んだ知識・ノウハウ | 対応方針、技術メモ |
| `episodes` | 過去の行動ログ | 「いつ何をしたか」の事実確認 |
| `procedures` | 手順書 | 「どうやるか」の手順確認 |
| `common_knowledge` | 全Anima共有の知識 | 組織ルール、システムガイド |
| `skills` | スキル・共通スキル（ベクトル検索） | スキルの発見・検索 |
| `activity_log` | 直近の行動ログ（ツール実行結果・メッセージ等） | 「さっき読んだメール」「先ほどの検索結果」など直近の事実確認 |
| `all` | 上記すべて（ベクトル検索 + activity_log BM25をRRFで統合） | キーワードの存在確認、広範な検索 |

---

## 権限がない

### 症状

- ツールを実行したら「権限がありません」「Permission denied」等のエラーが返された
- ファイルを読み書きしようとしたがアクセスできない
- コマンドを実行しようとしたが拒否された

### 原因

1. `permissions.json` で許可されていない操作（`permissions.md` のみの場合は `load_permissions` が読み込み時に JSON 相当へ正規化。無効 JSON は警告のうえ開放デフォルトにフォールバックすることもある）
2. 外部ツールのカテゴリがレジストリに未有効化（`check_permissions` の `available_but_not_enabled`）
3. ファイルパスが許可範囲外（保護ファイルへの書き込み、`file_roots` 外など）。加えて**グローバル**拒否は `permissions.global.json` とフレームワーク側パターンがある

### 対処手順

1. **自分の権限を確認する**
   ```
   check_permissions()
   ```
   - JSON で返る。`internal_tools`・`external_tools.enabled` / `available_but_not_enabled`・`file_access`（read/write）・`restrictions`（コマンド deny 等）を確認する
   - 生の設定は `read_memory_file(path="permissions.json")`（存在しない場合は `permissions.md`）で確認可能
   - システムプロンプトに注入される権限説明は、ランタイムが JSON から整形したテキストになる

2. **許可されている操作か確認する**
   - 自分の `anima_dir` 内は原則読み書き可能（`IDENTITY.md` / `SOUL.md` 等の保護ファイルは除く）。上司・同僚の `activity_log` や配下の `state/` 読み取りはロール次第で `check_permissions` の `file_access` に反映される
   - シェルコマンド: `permissions.json` の `commands`（allow/deny）に従う。グローバル危険パターンはフレームワーク側でもブロックされる

3. **権限が必要な場合の対応**
   - その操作が本当に必要か再検討する
   - 別のアプローチ（許可された範囲内の操作）で代替できないか考える
   - 代替不可能な場合は上司に権限追加を依頼する:
   ```
   send_message(
       to="上司の名前",
       content="【権限追加依頼】\n目的: XXXの作業のため\n必要な権限: /path/to/dir の読み取り\n理由: YYYの情報を参照する必要があるため",
       intent="question"
   )
   ```

4. **絶対にやってはいけないこと**
   - 権限チェックを回避しようとすること
   - 許可されていないコマンドを別の方法で実行しようとすること
   - 他のAnimaの権限を利用しようとすること

---

## ツールが使えない

### 症状

- ツールを呼び出したが「ツールが見つかりません」等のエラーが返された
- 外部ツール（Slack, Gmail 等）が利用できない

### 原因

1. そのツールが `permissions.json`（または読み込み時に正規化された MD 由来設定）で許可されていない、またはゲート付きアクションが明示許可されていない
2. スキルファイルが見つからない
3. 外部サービスの認証情報が設定されていない

### 対処手順

1. **スキルでツールの使い方を確認する**
   - `read_memory_file` でシステムプロンプトのスキルカタログに示されたパス（例: `skills/foo/SKILL.md`, `common_skills/bar/SKILL.md`）を指定し、手順の全文を取得する
   - B-mode で外部ツールが許可されている場合、`Bash: animaworks-tool <ツール> <サブコマンド>` で呼び出しが可能

2. **権限を確認する**
   ```
   check_permissions()
   ```
   - `external_tools.enabled`: この Anima のツールレジストリに載っている外部ツールカテゴリ（セッションに実際に渡るもの）
   - `external_tools.available_but_not_enabled`: フレームワークに実装はあるが、この Anima ではレジストリに入っていないカテゴリ。`permissions.json` の許可・ゲート付きアクション・実行モードとあわせて確認する

3. **利用できない場合**
   - `permissions.json` で該当ツール／アクションが許可されているか、認証情報（`shared/credentials.json` 等）があるかを確認する
   - それでも不可なら上司に依頼する（「なぜそのツールが必要か」を明記）

4. **MCP 統合モード（S/C/D/G: Claude Agent SDK / Codex CLI / Cursor Agent / Gemini CLI）の場合**
   - 組み込みツールはプレフィックスなしで利用可能（例: `send_message`）。見つからない場合はプロセス再起動が必要
   - 外部ツールは `read_memory_file` でスキル本文を読みCLI使用法を確認し、**Bash** 経由で `animaworks-tool <ツール> <サブコマンド>` を実行する（エージェントの Bash ツールを使用）
   - 長時間ツール（画像生成、ローカルLLM等）は `animaworks-tool submit` で非同期実行

5. **D-mode（Cursor Agent）特有のよくある問題**
   - **CLI が見つからない**: ホストに `cursor-agent` CLI がインストールされているか確認する
   - **認証エラー**: ターミナルで `agent login` を実行してログインする
   - **フォールバック**: 解決しない場合は `execution_mode` を `A` にするか、モデルを LiteLLM 経由（Mode A）に切り替えて運用する

6. **G-mode（Gemini CLI）特有のよくある問題**
   - **CLI が見つからない**: ホストに `gemini` CLI がインストールされているか確認する
   - **認証エラー**: `gemini auth login` を実行するか、環境変数 `GEMINI_API_KEY` を設定する
   - **フォールバック**: 解決しない場合は `execution_mode` を `A` にするか、モデルを LiteLLM 経由（Mode A）に切り替える。`gemini/` プレフィックスは Google プロバイダ向けに `google/` へリマップされる場合がある

7. **A-mode（LiteLLM）の場合**
   - 外部ツールは `read_memory_file` でスキル本文を読み使い方を確認し、**Bash** 経由で `animaworks-tool <ツール> <サブコマンド>` を実行する

8. **ツールがエラーを返す場合**
   - エラーメッセージを正確に記録する
   - 認証エラーの場合は上司に報告する（認証情報の設定は管理者の責務）
   - 一時的なタイムアウト・レート制限なら短時間待って再試行する（回数・間隔はツール実装・サーバー設定による）
   - 改善しない場合は依頼者に事実・試したことを報告する

ツール体系の全体像は `operations/tool-usage-overview.md` を参照。

---

## コンテキストが長くなりすぎた

### 症状

- セッションが長時間続いている
- 応答が遅くなってきた
- システムから「コンテキスト上限に近づいている」旨の通知がある

### 原因

- 長時間の作業や多数のツール呼び出しでコンテキストウィンドウが消費された
- 大量のファイル内容を読み込んだ

### 対処手順

1. **作業状態を短期記憶に保存する**（MUST）
   - 現在の作業状態を `shortterm/` に書き出す（チャットセッション時は `shortterm/chat/`）:
   ```
   write_memory_file(
       path="shortterm/chat/session_state.md",
       content="## 作業状態\n\n### 実行中のタスク\n- XXXの実装（50%完了）\n\n### 次のステップ\n1. YYYを完了する\n2. ZZZをテストする\n\n### 重要な中間結果\n- AAAの調査結果: BBB\n- CCCの設定値: DDD",
       mode="overwrite"
   )
   ```
   - ハートビートセッション時は `shortterm/heartbeat/session_state.md` を使用

2. **`state/current_state.md` を更新する**（MUST）
   ```
   write_memory_file(
       path="state/current_state.md",
       content="## 現在のタスク\n\nXXXの実装\n\n### 進捗\n- 50%完了\n- 次回はYYYから再開\n\n### メモ\n- 重要な発見事項をここに記載",
       mode="overwrite"
   )
   ```

3. **重要な知見は永続記憶に保存する**（SHOULD）
   - 作業中に得た知見は `knowledge/` に保存:
   ```
   write_memory_file(
       path="knowledge/xxx-findings.md",
       content="# XXXに関する知見\n\n## 発見事項\n...",
       mode="overwrite"
   )
   ```

4. **セッション継続を待つ**
   - システムが自動的に新しいセッションを開始する
   - 新セッションでは `shortterm/chat/`（または `shortterm/heartbeat/`）の内容がコンテキストに含まれる
   - `state/current_state.md` を読み直して作業を再開する

### 予防策

- 大きなファイルは全体を読まず、必要な部分だけ検索する
- 長い作業は定期的に `state/current_state.md` を更新する
- 中間結果はこまめに記憶に書き出す

---

## メッセージ送信でエラーが返った

### 症状と原因

- `RecipientResolutionError`、外部チャネルの `DeliveryFailed`、権限・会社境界エラーなど、実際の配送エラーが返る
- `send_message` で同一 run 内に同じ宛先へ2通目を送ろうとすると拒否される（重複防止）。宛先数の上限はない
- `post_channel` で同じ run 内に同じチャネルへ2回目を投稿しようとすると拒否される。run 間の投稿 cooldown はない
- 時間・日単位の送信予算や会話深度による送信拒否はない。内部 Anima 間の深度は診断ログに記録される場合がある

### 対処手順

1. **エラー内容を確認する**: `to` の解決先、intent（`report` / `question`）、チャネル ACL、会社境界、外部 API の応答を確認する
2. **重複送信を確認する**: 同一 run ですでに送った宛先には追加 DM を送れない。宛先数を理由に待つ必要はない
3. **配信失敗を調べる**: Slack / Chatwork 等の外部チャネルの場合は、返された配送エラーと接続設定を確認する
4. **Inbox の Provider エラーの場合**: `rate_guard` が指定する回復時間後に未読メッセージが再処理される

詳細は `communication/sending-limits.md` を参照。

---

## コマンドがブロックされた

### 症状

- コマンドを実行しようとしたら「PermissionDenied」「Command blocked」等のエラーが返された
- 特定のコマンドだけが実行できない

### 原因

1. フレームワーク／`permissions.global.json` のグローバル拒否パターンに該当するコマンド（例: `rm -rf /` 等）
2. `permissions.json` の `commands.deny` に列挙されたコマンド

### 対処手順

1. **自分の権限を確認する**
   ```
   check_permissions()
   ```
   - `restrictions` に deny されたコマンドが列挙される。併せて `read_memory_file(path="permissions.json")` で設定を直接確認（レガシー環境では `permissions.md`）

2. **代替手段を検討する**
   - ブロックされたコマンドと同等の操作を、許可されたツールで実現できないか考える
   - 例: `rm -rf` がブロックされている場合、個別ファイルの削除は許可されている可能性がある

3. **権限変更が必要な場合**
   - 上司にブロック解除を依頼する
   - 依頼時は「なぜそのコマンドが必要か」を明記すること

---

## プロンプトが短縮された

### 症状

- 通常よりシステムプロンプトが薄い、Priming（自動想起）が空に近い
- 長い会話や大きなユーザーメッセージのあと、応答前にプロンプトが再構築されたような挙動がある

### 原因

大きく 2 層ある。

**1. Priming（自動想起）のティア** — `core/prompt/builder.py` の `resolve_prompt_tier(context_window)` が、推定コンテキストウィンドウからティアを決める。ウィンドウの解決順は `core/prompt/context.py` `resolve_context_window`: **`~/.animaworks/models.json`（SSoT）** → 非推奨の `config.json` `model_context_windows` → `MODEL_CONTEXT_WINDOWS` 等のコード内フォールバック → 既定 128k。

| ティア | 条件（`context_window`） | Priming の扱い（`core/agent/priming.py`） |
|--------|--------------------------|---------------------------------------------|
| full | **≥ 128_000** | compact の通常取得を `priming.max_tokens` の範囲内で整形して載せる |
| standard | **≥ 32_000 かつ < 128_000** | 同じ compact 経路だが、取得予算を最大 1000 トークンに制限 |
| light | **≥ 16_000 かつ < 32_000** | compact の基本コンテキストを取得し、関連知識・エピソード検索を抑制 |
| minimal | **< 16_000** | compact の基本コンテキストを維持し、関連知識・エピソード検索を抑制 |

ハートビート／cron 用のクエリ文は、直近の `[REFLECTION]` を activity_log から集めたテキストになる（長いテンプレ全文ではない）。

**2. システムプロンプト本体の収縮** — `core/agent/priming.py` `_fit_prompt_to_context_window`: システム＋ユーザーの推定トークン + ツールスキーマ overhead が **コンテキストウィンドウの約 80%** を超えると、`build_system_prompt` を **システムバジェット 75% → 50% → 25%** と段階的に縮めて再構築する。**25% 以下の段**では **Priming ブロックと人間向け通知ブロックを空にして**から当てる。それでも収まらなければシステムプロンプトを**バイト単位でハードトランケート**する。

### 対処手順

1. **足りない文脈は明示取得する**: `search_memory` / `read_memory_file` で組織・手順・共有知識を読む（特に `minimal` / `light` では Priming が弱い）
2. **作業状態をディスクに残す**: `state/current_state.md` や `shortterm/` に要約を書いておき、セッションが切れても再開できるようにする
3. **上司・管理者に相談する**: 実運用で窮屈なら `models.json` の `context_window` やモデル変更を検討する

---

## その他のよくある問題

### ファイルが見つからない

- **原因**: パスの指定ミス、ファイルが存在しない
- **対処**: Mode S では `Glob`、それ以外は `search_memory` や既知パスの `read_memory_file` で当たる
- **注意**: `read_memory_file` は Anima ディレクトリ相対（例: `knowledge/xxx.md`）に加え、`common_knowledge/`・`reference/`・`common_skills/` プレフィックスで共有ディレクトリを読める。`Read`（エージェント組み込み）は別ルールでパスが決まる

### read_channel で inbox を指定できない

- **原因**: `read_channel` は Board の共有チャネル用。inbox（受信箱）はチャネルではない
- **対処**: inbox のメッセージはシステムが自動処理する。`read_channel` に `inbox` や `inbox/` を指定するとエラーになる

### コマンドがタイムアウトする

- **原因**: 処理時間が `timeout` を超えた
- **対処**: Bash 実行時の `timeout` パラメータを増やす（デフォルト: 30秒）
- **注意**: 長時間実行するコマンドには適切なタイムアウト値を設定すること

### 相手のAnimaが存在しない

- **原因**: Anima名の間違い、またはそのAnimaがまだ作成されていない
- **対処**: 上司に確認する。組織構造は `reference/organization/structure.md` を参照

### venv再構築後にSEGVが多発する

- **原因**: ChromaDB や PyTorch のメジャーバージョンが上がると、既存の vectordb（HNSW セグメント）と互換性がなくなり、読み取り時に SEGV（Segmentation Fault）が発生する。ChromaDB 1.5.x の Rust バインディングは破損したインデックスに対して Python 例外ではなく SEGV を起こす既知問題がある（chromadb #6852, #6949, #6979）
- **対処**: venv を再構築したら vectordb も必ず全再構築する
  1. サーバーを停止する
  2. `~/.animaworks/vectordb/` と各 `~/.animaworks/animas/*/vectordb/` を削除（またはバックアップ後削除）
  3. `~/.animaworks/index_meta.json` を削除
  4. サーバーを起動する（起動時に自動で再インデックスされる）
- **予防**: パッケージ更新時は chromadb・torch・sentence-transformers のバージョン変更を確認し、変更があれば vectordb を再構築する
