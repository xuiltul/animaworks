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

1. 相手の名前（Anima名）が間違っている
2. サーバーが停止している
3. 相手がハートビート間隔の合間にいる（次の起動まで未読のまま）
4. 送信処理自体がエラーで失敗していた
5. `intent` が未指定または不正（report / question のみ許可。タスク委譲は `delegate_task` を使用）
6. セッション内DM制限超過（同一宛先へは1回のみ、1セッションあたりの宛先数上限。ロールにより異なり、general は2人まで）

### 対処手順

1. **送信先の名前を確認する**
   - `send_message` の `to` パラメータに指定した名前が正しいか確認する
   - 名前は大文字・小文字を区別する。`identity.md` に記載された正式名称を使うこと
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
   - `state/current_state.md` にブロック理由として記載する
   - 上司に報告する

### 具体例

```
# 名前を間違えていた場合
send_message(to="Aoi", content="...", intent="report")   # OK
send_message(to="aoi", content="...", intent="report")  # 名前が異なればエラーになる可能性あり

# DM は intent 必須（report / question のみ）
# 1セッションあたりの宛先数はロールにより異なる（general は2人まで）。同一宛先へは1回のみ
send_message(
    to="aoi",
    content="了解しました。作業を開始します。",
    intent="report",           # 必須: report / question
    reply_to="msg-abc123",     # 任意: 元メッセージのID
    thread_id="thread-xyz789"  # 任意: スレッドID
)

# 確認・お礼・お知らせのみのDMは不可 → post_channel（Board）を使用
```

---

## タスクがブロックされた

### 症状

- 作業を進めようとしたが、必要な情報や権限が不足している
- 他のAnimaの作業完了を待っている状態
- 外部サービスがエラーを返す

### 原因

1. 依存タスクが未完了
2. 権限不足（permissions.json で許可されていない操作を実行しようとした）
3. 必要な情報が不足している
4. 外部サービスの障害

### 対処手順

1. **ブロック内容を明確化する**
   - 何が足りないのかを具体的に特定する
   - 「誰の」「何の作業が」「いつまでに」必要かを整理する

2. **`state/current_state.md` を更新する**
   ```
   write_memory_file(
       path="state/current_state.md",
       content="## 現在のタスク\n\nXXXの実装\n\n### ブロック中\n- 原因: YYYの作業完了待ち\n- 待ち先: ZZZさん\n- 発生日時: 2026-02-15 10:00",
       mode="overwrite"
   )
   ```

3. **自力で解決できるか判断する**
   - 別のアプローチで回避できないか検討する
   - 記憶を検索して過去に同様の問題がなかったか確認する:
     ```
     search_memory(query="ブロック", scope="episodes")
     search_memory(query="回避", scope="knowledge")
     ```

4. **解決できない場合はエスカレーションする**（`troubleshooting/escalation-flowchart.md` 参照）
   - 上司に報告する。報告には以下を含めること:
     - 何をしようとしたか
     - 何にブロックされているか
     - いつからブロックされているか
     - 自分で試みた対処
   ```
   send_message(
       to="上司の名前",
       content="【ブロック報告】\nタスク: XXXの実装\nブロック原因: YYYのAPI権限が不足\n発生: 2026-02-15 10:00\n試行: permissions.jsonを確認したが該当設定なし\n依頼: API権限の追加をお願いします",
       intent="report"
   )
   ```

5. **ブロック中でも進められる作業がないか確認する**
   - 永続タスクキュー（`Bash: animaworks-tool task list`）および `state/pending/` 配下のタスクに他の作業がないか確認する
   - ブロックされていない別のタスクに着手する

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
   - 検索でヒットしない場合は `Glob` でディレクトリの内容を一覧する。`path` を省略すると anima_dir のルートが表示され、knowledge/, procedures/, episodes/ 等のサブディレクトリが確認できる
   - ファイル名から目的のファイルを見つけて直接読む:
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
| `all` | 上記すべて | キーワードの存在確認、広範な検索 |

---

## 権限がない

### 症状

- ツールを実行したら「権限がありません」「Permission denied」等のエラーが返された
- ファイルを読み書きしようとしたがアクセスできない
- コマンドを実行しようとしたが拒否された

### 原因

1. `permissions.json` で許可されていない操作を実行しようとした
2. 外部ツールのカテゴリが未有効化
3. ファイルパスが許可範囲外

### 対処手順

1. **自分の権限を確認する**
   ```
   check_permissions()
   ```
   - 利用可能な内部ツール・外部ツール・ファイルアクセス・制限事項が一覧で返る
   - 詳細は `read_memory_file(path="permissions.json")` で確認可能
   - `permissions.json` の主なセクション:
     - 「ファイル操作」「読める場所」: 読み取り可能なパス
     - 「コマンド実行」「実行できるコマンド」: 実行可能なコマンドのホワイトリスト
     - 「実行できないコマンド」: ブロック対象コマンド
     - 外部ツール: permissions.json で許可されたカテゴリが有効化される

2. **許可されている操作か確認する**
   - 自分の anima_dir 内は読み書き可能。共有ディレクトリ・部下の管理ファイル等は `check_permissions` で確認
   - コマンド: 「実行できるコマンド」に列挙されたコマンドのみ実行可能

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

1. そのツール自体が `permissions.json` で許可されていない
2. スキルファイルが見つからない
3. 外部サービスの認証情報が設定されていない

### 対処手順

1. **スキルでツールの使い方を確認する**
   - `skill` ツールでスキル名を指定し、手順の全文を取得する。スキル一覧はツール説明の `<available_skills>` ブロックに表示される
   - B-mode で外部ツールが許可されている場合、`Bash: animaworks-tool <ツール> <サブコマンド>` で呼び出しが可能

2. **権限を確認する**
   ```
   check_permissions()
   ```
   - `external_tools.enabled` に現在有効なカテゴリ、`external_tools.available_but_not_enabled` に許可済みだが未有効のカテゴリが返る
   - permissions.json で許可されていないカテゴリは使用できない

3. **カテゴリが許可されていない場合**
   - 上司に利用許可を依頼する
   - 依頼時は「なぜそのツールが必要か」を明記すること

4. **MCP 統合モード（S/C/D/G: Claude Agent SDK / Codex CLI / Cursor Agent / Gemini CLI）の場合**
   - 組み込みツールはプレフィックスなしで利用可能（例: `send_message`）。見つからない場合はプロセス再起動が必要
   - 外部ツールは `skill` ツールでCLI使用法を確認し、**Bash** 経由で `animaworks-tool <ツール> <サブコマンド>` を実行する（エージェントの Bash ツールを使用）
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
   - 外部ツールは `skill` で使い方を確認し、**Bash** 経由で `animaworks-tool <ツール> <サブコマンド>` を実行する

8. **ツールがエラーを返す場合**
   - エラーメッセージを正確に記録する
   - 認証エラーの場合は上司に報告する（認証情報の設定は管理者の責務）
   - タイムアウトの場合はリトライする（最大3回まで）
   - リトライでも解決しない場合はブロックとして報告する

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

## メッセージ送信が制限された

### 症状

- `send_message` や `post_channel` を実行したらエラーが返された
- `GlobalOutboundLimitExceeded: 1時間あたりの送信上限（N通）に到達しています...` 等のメッセージが表示された
- `ConversationDepthExceeded: {相手}との会話が10分間に6ターンに達しました...` と表示された

### 原因

- **ロール別制限**: 1時間あたり・24時間あたりの送信上限は `status.json` の `role` に応じたデフォルト値が適用される（例: general 15/50通、manager 60/300通）。`status.json` の `max_outbound_per_hour` / `max_outbound_per_day` で個別に上書き可能
- 同一チャネルへの連続投稿がクールダウン期間内だった（`config.json` の `heartbeat.channel_post_cooldown_s`、デフォルト300秒）
- 2者間のDM往復が深度制限（10分間に6ターン）を超えた（`heartbeat.depth_window_s` / `heartbeat.max_depth`）

### 対処手順

1. **エラーメッセージを確認する**: 時間制限・24時間制限・深度制限のいずれかを特定する
2. **送信履歴を振り返る**: 不要な送信がなかったか確認する
3. **待機する**: 時間制限なら次の1時間枠まで、24時間制限なら翌日まで、深度制限なら次のハートビートサイクルまで待つ
4. **送信内容を記録する**: このターンでは `send_message` を使わず、送信したい内容を `state/current_state.md` に記録し、次のセッションで送信する
5. **緊急連絡**: `call_human` は制限対象外なので、人間への連絡は引き続き可能
6. **送信を統合する**: 複数の報告を1通にまとめる。深度制限に達した場合は、複雑な議論を Board チャネルに移行する

詳細は `communication/sending-limits.md` を参照。

---

## コマンドがブロックされた

### 症状

- コマンドを実行しようとしたら「PermissionDenied」「Command blocked」等のエラーが返された
- 特定のコマンドだけが実行できない

### 原因

1. システム全体のブロックリストに含まれるコマンド（`rm -rf /` 等の危険なコマンド）
2. `permissions.json` の「実行できないコマンド」セクションに記載されたコマンド

### 対処手順

1. **自分の権限を確認する**
   ```
   read_memory_file(path="permissions.json")
   ```
   - `## 実行できないコマンド` セクションにブロック対象が記載されている

2. **代替手段を検討する**
   - ブロックされたコマンドと同等の操作を、許可されたツールで実現できないか考える
   - 例: `rm -rf` がブロックされている場合、個別ファイルの削除は許可されている可能性がある

3. **権限変更が必要な場合**
   - 上司にブロック解除を依頼する
   - 依頼時は「なぜそのコマンドが必要か」を明記すること

---

## プロンプトが短縮された

### 症状

- 通常表示されるはずの情報（組織コンテキスト、記憶ガイド等）がシステムプロンプトに含まれていない
- ツールの種類が少ない
- 記憶の自動想起（Priming）が動作していないように見える

### 原因

コンテキストウィンドウが小さいモデルを使用している場合、システムプロンプトが段階的に縮小される（Tiered System Prompt）。
`status.json` のモデル名からコンテキストウィンドウを推定し、`~/.animaworks/models.json` または `config.json` の `model_context_windows` でオーバーライド可能。

| ティア | コンテキストウィンドウ | 省略される情報 |
|--------|----------------------|--------------|
| T1 (FULL) | 128k+ トークン | なし（全情報を表示） |
| T2 (STANDARD) | 32k〜128k トークン | 蒸留知識（廃止予定）・Priming バジェット縮小 |
| T3 (LIGHT) | 16k〜32k トークン | bootstrap, vision, specialty, 蒸留知識, 記憶ガイド 省略 |
| T4 (MINIMAL) | 16k 未満 トークン | permissions, Priming, org, messaging, emotion も省略 |

### 対処手順

1. **必要な情報は自分で検索する**: 省略された情報は `search_memory` や `read_memory_file` で明示的に取得する
2. **上司に相談する**: モデルの変更が必要な場合は上司に依頼する

---

## その他のよくある問題

### ファイルが見つからない

- **原因**: パスの指定ミス、ファイルが存在しない
- **対処**: `Glob` でディレクトリ内容を確認してからパスを指定する
- **注意**: `read_memory_file` は Anima ディレクトリからの相対パス（例: `knowledge/xxx.md`, `reference/organization/structure.md`）。`Read` は絶対パスを使用する

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
