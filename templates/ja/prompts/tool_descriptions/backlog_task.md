タスクキューに新しいタスクを追加する。人間からの指示は必ず source='human' で記録すること。Anima間の委任は source='anima' で記録する。
【書く前に読む（MUST）】追加前に `list_tasks` を読み、同じ PR / Issue / 対象の未完タスクが無いか確認する。あれば追加しない。
【書いた後に読む（MUST）】追加後に `list_tasks` で読み戻し、summary（先頭に `[PR #N]` 等）を確認する。
