# Zoom RTMS 会議リスナー セットアップガイド

AnimaWorks が Zoom 会議のトランスクリプト（話者ラベル付き文字起こし）を **Zoom RTMS (Real-Time Media Streams)** でリアルタイム取り込みし、anima の inbox へチャンク注入するための運用セットアップ手順。

> このガイドは取り込みパイプライン（Zoom → inbox 注入）のセットアップを対象とする。会議内容を anima がどう扱うか（レポート作成・承認・エスカレーション）を規定するワークフロー（スキル）は別途配備する。本ガイドでゴールとするのは「対象 anima の inbox にトランスクリプトのチャンクが届くこと」まで。

## 概要

RTMS は会議の音声/映像/**話者ID+timestamp付きトランスクリプト**を WebSocket で配信する Zoom 公式機能。**bot 参加は不要**で、対象アカウントが会議をホストした瞬間に webhook が飛び、ストリーム接続が始まる。AnimaWorks は **transcript ストリームのみ**を受信し、音声（audio raw）は取得しない。会議中の発言・チャット投稿も一切行わない（**聞くだけ**）。

```
[Zoom 会議] ──meeting.rtms_started webhook──▶ [POST /api/webhooks/zoom]
                                                       │
                                                       ▼
[Zoom RTMS signaling/media WS] ◀──HMAC署名ハンドシェイク── [ZoomRTMSManager (server/zoom_gateway.py)]
        │ transcript発話（文単位）                          │ 発話をバッファ
        └────────────────────────────────────────────────▶ │ chunk_interval_seconds 経過 or
                                                       │    chunk_max_chars 到達でフラッシュ
                                                       ▼
                              Messenger.receive_external(source="zoom")
                                                       │
                                                       ▼
                              shared/inbox/{anima}/{id}.json  ──▶  anima が通常サイクルで処理
```

会議終了時（`meeting.rtms_stopped`）は残バッファをフラッシュした後、`intent="meeting_ended"` のトリガメッセージを 1 通注入する（発話ゼロの会議では注入しない）。

## 前提条件

- AnimaWorks サーバーが公開 HTTPS で到達可能なこと（webhook 受信のため）。Slack Socket Mode と異なり、Zoom は **HTTP webhook でイベントを push** するため、NAT 内サーバーでは公開エンドポイント（リバースプロキシ / トンネル）が必要。
- ホストとなる Zoom アカウント（会議を主催するユーザー）。**ホスト自身の Zoom プランは Basic（無料）で可**。
- Zoom Marketplace 上の非公開 General App（下記で作成。自アカウント利用なら審査不要）。
- **Developer Pack（Zoom Build Platform クレジット）** の契約。RTMS の分課金はこれが唯一の有料要件（下記 3-5）。
- 会議参加者の Zoom クライアントが **6.5.5 以上**であること（RTMS 対象要件。未満のクライアントの発話は取り込まれない）。
- `websockets` ライブラリ（RTMS プロトコルの自前実装で使用。`pip install "animaworks[communication]"` に含まれる）。

---

## 1. Zoom App の作成（Zoom App Marketplace）

RTMS は **General App** でのみ利用できる。Server-to-Server OAuth や Webhook-only App では RTMS が使えない。

1. https://marketplace.zoom.us/ にホストアカウントでログイン
2. 右上「Develop」→「Build App」
3. アプリ種別で **General App** を選択
4. アプリ名を入力（社内利用のため任意。例: `AnimaWorks Meeting Listener`）
5. **配布設定は非公開（"Manage this app only for my account" / account-level）**にする。自アカウント内利用のみなら Marketplace の公開審査は不要。
6. 「Basic Information」で以下を控える:

| 項目 | 対応する `.env` キー | 用途 |
|------|---------------------|------|
| Client ID | `ZOOM_CLIENT_ID` | RTMS ハンドシェイク署名に使用 |
| Client Secret | `ZOOM_CLIENT_SECRET` | RTMS ハンドシェイク署名（HMAC鍵）に使用 |
| Secret Token | `ZOOM_SECRET_TOKEN` | webhook 検証（`x-zm-signature` / URL validation）に使用 |

> Client Secret と Secret Token は再表示できない場合がある。作成直後に安全な場所へ控えること。漏洩時はローテーション（再生成）する。

## 2. RTMS 機能とスコープの付与

1. アプリ設定で **Realtime Media Streams (RTMS) を有効化**する（"Add RTMS features to your app" / Features → RTMS のトグル）。
2. 「Scopes」で以下を付与する。スコープ名は Marketplace のスコープピッカーで表記が **複数形/単数形どちらか**になっていることがある（`meeting:read:meeting_transcripts` と `meeting:read:meeting_transcript`）。**ピッカーに出てくる方**を選ぶ。

| スコープ | 用途 | 必須 |
|----------|------|------|
| `meeting:read:meeting_transcripts` | 会議トランスクリプト（文字起こし）の受信 | 必須 |
| `rtms:read:rtms_started` | `meeting.rtms_started` イベント受信 | 必須 |
| `rtms:read:rtms_stopped` | `meeting.rtms_stopped` イベント受信 | 必須 |

> **audio / video スコープ（`meeting:read:meeting_audio` 等）は付与しない。** AnimaWorks は transcript のみ受信する設計であり、音声スコープを付けると課金対象・データ受信量が増える。

> RTMS 機能のトグルやスコープが Marketplace 画面に**表示されない**場合、アカウントで RTMS のバックエンド有効化がまだ済んでいない可能性がある（後述の 3-5 / トラブルシューティング参照）。

## 3. Webhook（Event Subscription）の設定

1. アプリ設定「Feature」→「Event Subscriptions」（または「Access」→ Webhook）を有効化
2. **Event notification endpoint URL** に AnimaWorks の webhook を設定:

   ```
   https://<あなたのサーバー>/api/webhooks/zoom
   ```

   （AnimaWorks 側の実装エンドポイントは `POST /api/webhooks/zoom`。ルーターは `server/routes/webhooks.py`、Chatwork webhook と同じ mount 配下）
3. **URL validation**: Zoom は endpoint 登録時に `endpoint.url_validation` イベントを送る。ペイロードの `plainToken` を **Secret Token（`ZOOM_SECRET_TOKEN`）を鍵とした HMAC-SHA256** でハッシュし、`{"plainToken": ..., "encryptedToken": <hex>}` を返す必要がある。AnimaWorks の webhook ハンドラがこれを自動処理するため、**サーバーを先に起動して `.env` に `ZOOM_SECRET_TOKEN` を設定した状態**で「Validate」を押すこと。
4. 「Add Events」で以下 2 つを購読:

| イベント | 意味 |
|----------|------|
| `meeting.rtms_started` | RTMS 対象会議が開始。signaling WS の接続情報がペイロードに含まれる |
| `meeting.rtms_stopped` | 会議終了 / RTMS 停止 |

5. 以降のイベント配信は各リクエストの `x-zm-signature` ヘッダ（Secret Token による HMAC-SHA256）で検証される。**署名不正のリクエストは 401 で拒否**される。

## 4. auto-start（自動開始）の有効化

固定会議室での無人運用には、対象アカウントが会議をホストした瞬間に RTMS を自動開始させる。

1. Zoom Web ポータルの **Settings（設定）→ Zoom Apps**（アカウント / 管理者設定）を開く
2. **「Auto-start apps that access shared realtime meeting content」** の項目で「Choose an app to auto-start（自動開始するアプリを選択）」をクリック
3. 作成した General App を選択して有効化

有効化後、対象アカウントが会議をホストすると `meeting.rtms_started` webhook が飛ぶ。RTMS の開始は参加者に通知される（同意まわりは Zoom 側の機能で担保）。

> アプリが選択肢（ドロップダウン）に**出てこない**、または選択しても `meeting.rtms_started` が飛ばない場合は、アカウントでの RTMS バックエンド有効化待ちのことが多い（3-5 参照）。

## 5. Developer Pack の契約（課金の分かれ目・最優先確認事項）

RTMS の分課金は **Zoom Build Platform クレジット（1 credit = $1）** で支払う。**契約プランで総額が約 10 倍変わる**ため、契約前に必ず実レートを確認する。

### 契約前チェック（最優先）

1. https://zoom.us/pricing/developer （Developer / Build Platform 料金ページ）を開く
2. **Pay as You Go（従量後払い）** が選べるか、その **実レートと最低下限（月額ミニマム）の有無**を確認する
3. 下限のない Pay as You Go が使えるなら、それを選ぶ（週2回×1時間程度の利用では従量が圧倒的に安い）

| プラン | 実質コスト | 備考 |
|--------|-----------|------|
| **Pay as You Go（従量）** | 実消費のみ（週2×1h想定で **月$10前後**） | 2026/5 に self-service 追加。**下限の有無を必ず確認** |
| 月額 100 credit | **$100/月（下限）** | 実消費が少なくても $100 発生する |
| 月額 500 credit | $450/月 | 大規模利用向け |

### 単価（Zoom 公表値ベース）

- RTMS 基本レート: **$0.01 / 会議ストリーミング分**
- transcript（文字起こしアドオン）上乗せ: **~$0.01 / 分**（合計 **~$0.02 / 分**）

> **注意**: 月額クレジットプラン（$100/月下限）を誤って選ぶと、実消費が月$10でも $100/月 の固定費になる。**Pay as You Go の実レートと下限有無の確認がセットアップ最優先事項**。未使用クレジットの翌月繰越は契約依存で保証されない。

## 6. 参加者・会議設定の要件

- 参加者の Zoom クライアントは **6.5.5 以上**。未満の参加者の発話は RTMS 対象外となり欠落する。
- **文字起こし言語を日本語に設定**する。RTMS は Zoom 側の文字起こし（ライブトランスクリプト / キャプション）設定に従うため、言語が誤設定だと文字起こし品質が落ちる。会議設定またはアカウント設定でトランスクリプト言語を「日本語」に固定しておくこと（実装側では言語補正しない）。

---

## 7. AnimaWorks 側のセットアップ

### 7-1. `.env` 設定

認証情報は **`.env` のみ**に置く（config ファイルには置かない）。1 の手順で控えた値を設定:

```bash
ZOOM_CLIENT_ID=<Basic Information の Client ID>
ZOOM_CLIENT_SECRET=<Basic Information の Client Secret>
ZOOM_SECRET_TOKEN=<Basic Information の Secret Token>
```

| キー | 用途 |
|------|------|
| `ZOOM_CLIENT_ID` | RTMS signaling/media ハンドシェイクの HMAC 署名（`client_id,meeting_uuid,rtms_stream_id` を Client Secret で署名） |
| `ZOOM_CLIENT_SECRET` | 同上ハンドシェイク署名の HMAC 鍵 |
| `ZOOM_SECRET_TOKEN` | webhook の URL validation（`plainToken`→`encryptedToken`）と `x-zm-signature` 検証 |

### 7-2. config 設定

`~/.animaworks/config.json` の `external_messaging.zoom` で制御する（`ZoomRTMSConfig`）。

```json
{
  "external_messaging": {
    "zoom": {
      "enabled": true,
      "default_anima": "kotoha",
      "meeting_mapping": {
        "8912345678": "kotoha"
      },
      "chunk_interval_seconds": 300,
      "chunk_max_chars": 4000
    }
  }
}
```

| キー | 型 | デフォルト | 説明 |
|------|----|-----------|------|
| `enabled` | bool | `false` | Zoom RTMS 取り込みの有効/無効 |
| `default_anima` | string | `""` | `meeting_mapping` に無い会議のフォールバック先 anima。ここも空だとチャンクは破棄され WARN ログのみ |
| `meeting_mapping` | object | `{}` | **会議ID（meeting_id）→ anima 名**。会議ごとに聴かせる anima を指定 |
| `chunk_interval_seconds` | int | `300` | チャンクをフラッシュする時間間隔（秒）。5分粒度が要約・把握の実用単位 |
| `chunk_max_chars` | int | `4000` | 1 チャンクの最大文字数。時間間隔より先にこれに達したらフラッシュ |

- **会議ID（meeting_id）の確認方法**: Zoom の会議 URL または招待に含まれる数字列（例: `https://zoom.us/j/8912345678` → `8912345678`）。固定会議室（Personal Meeting ID / 定例会議）の ID を指定する。
- `meeting_mapping` に無い会議は `default_anima` にルーティングされる。両方空だと破棄される（クラッシュはしない）。

### 7-3. サーバー再起動

`.env` と config を設定したらサーバーを再起動:

```bash
animaworks start
```

起動ログに Zoom ゲートウェイの起動が出れば OK（`enabled: false` または未設定なら disabled ログのみ）。

### 7-4. health 確認

`/api/system/health` 応答の `zoom_gateway` フィールドで稼働状態と webhook 最終受信時刻を確認できる:

```bash
curl -s https://<あなたのサーバー>/api/system/health | jq '.zoom_gateway'
```

`enabled: true` かつゲートウェイが起動していれば health に現れる。webhook が一度も届いていない場合は最終受信時刻が空になる（後述トラブルシューティングの切り分けに使う）。

---

## 8. 動作確認

1. **テスト会議を開催**: ホストアカウントで対象会議室（`meeting_mapping` に登録した ID）の会議を開始する。参加者は Zoom クライアント 6.5.5 以上を使う。
2. **webhook 受信を確認**: サーバーログに `meeting.rtms_started` 受信と signaling/media WS 接続のログが出ることを確認。`/api/system/health` の `zoom_gateway` の最終受信時刻が更新される。
3. **発話してみる**: 会議で数分間、複数人で会話する（文字起こしが日本語で出ることを Zoom のライブキャプションでも確認できると良い）。
4. **チャンク注入を確認**: `chunk_interval_seconds`（デフォルト5分）経過、または `chunk_max_chars`（4000文字）到達で、対象 anima の inbox にチャンク JSON が届く:

   ```bash
   ls -lt ~/.animaworks/shared/inbox/kotoha/
   ```

   中身は `source="zoom"`, `intent="meeting_transcript"` で、本文先頭に定型ヘッダ（`[Zoom会議実況 チャンク#N | 会議: {topic} ({meeting_id}) | ...]`）と話者ラベル付きの発話行が入る。
5. **会議終了トリガを確認**: 会議を終了すると残バッファがフラッシュされ、最後に `intent="meeting_ended"` のメッセージ（`[Zoom会議終了 | ...]`）が 1 通注入される。発話が一度も無かった会議では注入されない。

---

## 9. 月次コスト見積式

```
月額コスト（USD） ≒ 月間の会議ストリーミング総分数 × $0.02
```

- `$0.02/分` = RTMS 基本 `$0.01/分` + transcript 加算 `~$0.01/分`
- 例: 週2回 × 1時間 × 月4週 = 480分/月 → **480 × $0.02 ≒ $9.6/月**
- ただし **月額クレジットプラン（$100/月下限）を選んでいると実消費に関わらず $100/月**。Pay as You Go（従量・下限なし）を選ぶこと（5 参照）。
- 同時開催や参加時間の長い会議が増えると分数が伸びる。auto-start は**対象会議室のみ**に限定し、不要な会議まで取り込まないこと。

---

## 10. トラブルシューティング

### `meeting.rtms_started` が飛ばない / アプリが auto-start ドロップダウンに出ない

**最も多い症状。** 設定（webhook 検証済み・イベント購読・スコープ）が全て正しくても、アカウントで **RTMS バックエンドが有効化されていない**と発火しない（`meeting.started`/`meeting.ended` は届くのに `rtms_started` だけ来ない、という切り分けになる）。

- 切り分け: `meeting.started` は届くか？ → 届くが `rtms_started` が来ないなら RTMS バックエンド未有効化を疑う。
- 対処: Zoom Developer Forum の RTMS カテゴリで **App ID / Client ID を提示して RTMS 有効化を依頼**する（開発アカウントでは Zoom 側の手動有効化が必要なケースが報告されている）。Marketplace に RTMS トグルやスコープ自体が出ない場合も同根。

### webhook が届かない（URL validation が通らない / イベントが来ない）

- 公開 URL `https://<サーバー>/api/webhooks/zoom` が外部から HTTPS で到達可能か（`curl` で外部から叩いて確認）。
- URL validation 失敗 → `.env` の `ZOOM_SECRET_TOKEN` が Basic Information の Secret Token と一致しているか、サーバーが token 設定済みで起動しているか。
- 署名 401 → 同じく `ZOOM_SECRET_TOKEN` の不一致。サーバーログの署名検証エラーを確認。
- `/api/system/health` の `zoom_gateway` 最終受信時刻が更新されないなら、そもそもリクエストがサーバーに届いていない（リバースプロキシ / ファイアウォール / トンネルを確認）。

### WS が接続直後に切れる / ハンドシェイク失敗

- `ZOOM_CLIENT_ID` / `ZOOM_CLIENT_SECRET` の誤り。RTMS ハンドシェイクは `HMAC-SHA256(client_id + "," + meeting_uuid + "," + rtms_stream_id, client_secret)` の署名で認証されるため、どちらかがずれると弾かれる。
- ホストが会議を離れて RTMS 権限の無い別ホストに切り替わると Zoom が WS を閉じる。新しい `meeting.rtms_started` を待って再接続する設計。
- keep-alive（約65秒）タイムアウトでの切断は、保持している rtms_started ペイロードで再ハンドシェイク（指数バックオフ最大5回）される。復旧不能時は次チャンクに `[接続断により一部欠落]` が注記される。ログで再接続回数を確認。

### チャンクが inbox に注入されない

- 発話が起きているか（無音・発話ゼロの会議では `meeting_transcript` チャンクも `meeting_ended` も注入されない、INFO ログのみ）。
- 会議の文字起こし言語設定（6 参照）。トランスクリプト自体が Zoom 側で生成されていないと RTMS に流れてこない。Zoom のライブキャプションが会議中に出ているか確認。
- `meeting_mapping` の会議 ID と実際の会議 ID が一致しているか。不一致で `default_anima` も空なら破棄され WARN ログが出る。
- フラッシュ条件未達: まだ `chunk_interval_seconds` 経過も `chunk_max_chars` 到達もしていないだけの可能性。短時間の会議では会議終了時のフラッシュを待つ。
- 参加者クライアントが 6.5.5 未満で RTMS 対象外になっていないか。

### 会議は取り込めているが anima が反応しない

- 取り込み（inbox 注入）と anima の処理は別レイヤー。inbox にファイルが届いていれば取り込みは成功。処理は anima の次の run サイクル（heartbeat/cron/手動）に依存する。
- `source="zoom"` + intent 付きメッセージは inbox watcher のクールダウンをバイパスして即処理対象になる。それでも動かない場合は anima 側のログを確認（本ガイドの範囲外）。

---

## 関連ファイル

| ファイル | 役割 |
|----------|------|
| `server/zoom_gateway.py` | `ZoomRTMSManager`（signaling/media WS 接続、HMAC ハンドシェイク、transcript 購読、keep-alive、発話バッファ、チャンクフラッシュ、`receive_external` 注入、再接続） |
| `server/routes/webhooks.py` | `POST /api/webhooks/zoom`（URL validation 応答、`x-zm-signature` 検証、`meeting.rtms_started`/`meeting.rtms_stopped` の Manager 委譲） |
| `server/app.py`（lifespan 付近） | `ZoomRTMSManager` の起動・停止 |
| `server/reload_manager.py` | Zoom 設定の reload 対応 |
| `server/routes/system.py` | health 応答への `zoom_gateway` 追加 |
| `core/config/schemas.py` | `ZoomRTMSConfig`（`external_messaging.zoom`） |
| `core/schemas.py` | `EXTERNAL_PLATFORM_SOURCES`（`"zoom"`） |
| `core/messenger.py` | `receive_external()` — inbox 配置・source_message_id dedup |

## 参照

- [Getting started with Realtime Media Streams — Zoom Developer Docs](https://developers.zoom.us/docs/rtms/meetings/getting-started/)
- [Get meeting transcripts using native WebSockets with RTMS — Zoom Developer Blog](https://developers.zoom.us/blog/realtime-mediastreams-websockets/)
- [Zoom Developer / Build Platform pricing](https://zoom.us/pricing/developer)
- Zoom Developer Forum「Realtime Media Streams」カテゴリ（RTMS バックエンド有効化の依頼先）
