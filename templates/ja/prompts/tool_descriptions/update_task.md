タスクのステータスを更新する。完了時は status='done'、取り下げ時は status='cancelled'。status='in_progress' は、実行中の TaskExec が途中経過を summary に書くときに使う。続きをやるときは `submit_tasks` で同じ task_id を再投入する。
【書く前に読む】更新前に `list_tasks` で現在の status / summary を読む（別 worker が既に done / cancelled にしていないか）。同じ対象のタスクが 2 つ以上あれば、新しい方を続けて古い方を cancelled にする。
【書いた後に読む】更新後に `list_tasks` で反映を確認する。修正を push して返信したらそれが完了なので、その時点で status='done' を書く。
