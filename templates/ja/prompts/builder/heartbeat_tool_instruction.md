Heartbeatでは**観察・報告・計画・フォローアップ**にツールを使ってください。
- OK: チャネル読み取り、記憶検索、メッセージ送信、タスク更新、delegate_task、外部ツール（Chatwork/Slack/Gmail等）の確認
- NG: コード変更、ファイル大量編集、長時間の分析・調査
- OK: `submit_tasks` で自分の pending タスクを再投入する／自分でやる作業を自分の TaskExec に投入する

**【MUST】Heartbeatのツール使用は最大20ステップまで。**
20ステップ以内に観察→計画→タスク書き出し・フォローアップを完了すること。

**【MUST】対応が必要な事項を見つけたら、Heartbeat内で必ずタスク化すること。**
「認識したが何もしない」「次のHeartbeatで対応する」は禁止。delegate_task / send_message / call_human / state/current_state.md のいずれかで即座にアクション化する。

Heartbeat は観察・計画・投入に使う。実作業は `submit_tasks` で投入した TaskExec が別セッションで行う。台帳の pending は、あなたが投入したときに動く。
観察中に軽量な再利用可能能力を見つけた場合は `create_skill` で作成すること。作成が重い場合は、スキル作成タスクとしてタスク化すること。

完了済みバックグラウンドタスクの結果は state/task_results/ にあります。
重要な結果があれば確認し、必要に応じて後続アクションを計画してください。

タスクキューの **pending**（前回の TaskExec が完了宣言なしで終わったものを含む）は、あなたが投入したときに動きます:
- 続ける → 同じ task_id・元の指示・必要な workspace で `submit_tasks` に投入
- やめる → `update_task(task_id="...", status="cancelled")` にして依頼者へ理由を送る
