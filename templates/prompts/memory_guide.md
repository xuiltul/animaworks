## あなたの記憶（書庫）

全ての記憶は `{anima_dir}/` にあります。

| ディレクトリ | 種類 | 内容 |
|-------------|------|------|
| `{anima_dir}/episodes/` | エピソード記憶 | 過去の行動ログ（日別） |
| `{anima_dir}/knowledge/` | 知識 | 学んだこと・対応方針・ノウハウ |
| `{anima_dir}/procedures/` | 手順書 | 作業の進め方 |
| `{anima_dir}/skills/` | スキル | 実行可能な能力・テンプレート付き手順 |
| `{anima_dir}/state/` | 現在の状態 | 今何をしているか |
| `{anima_dir}/shortterm/` | 短期記憶 | セッション引き継ぎ用の作業状態（一時的） |

知識ファイル: {knowledge_list}
エピソード: {episode_list}
手順書: {procedure_list}
スキル: {skill_names}

## 共有ユーザー記憶

全社員で共有するユーザー情報は `shared/users/` にあります。
ユーザーごとにサブディレクトリが存在し、以下のファイルで構成されます:

- `shared/users/{{ユーザー名}}/index.md` — 構造化プロフィール（基本情報・好み・注意事項）
- `shared/users/{{ユーザー名}}/log.md` — やりとりの時系列ログ（最新20件）

メッセージの送信者に対応するディレクトリがあれば、まず `index.md` を読んでください。
詳細な経緯が必要な場合のみ `log.md` を検索してください。

登録済みユーザー: {shared_users_list}
