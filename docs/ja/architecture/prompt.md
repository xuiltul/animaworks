> 確認したコミット: b304b7dc

# System prompt の構築

`core/prompt/builder.py` が記憶、実行モード、trigger、現在の依頼をもとに system prompt を構築し、`core/prompt/assembler.py` が section を XML boundary tag 付きで組み立てて budget に収める。固定部分を先頭にまとめ、turn ごとに変わる状態情報を後ろに置く。

## section の構成と順序

builder は6つの論理グループを作る。組み立て順は、グループ1、2、4、5、6、最後に動的なグループ3である。

1. **環境と行動規則** — runtime 情報、workspace、Anima の `IDENTITY.md` / `SOUL.md`、`AGENTS.md`、behavior rules。
2. **Anima 自身の情報** — bootstrap 状態、会社の vision、speciality、permissions。
3. **記憶と能力** — memory guide、trigger / mode に合わせた tool guide、利用可能な外部 tool、active skill context、skill catalog。
4. **組織とコミュニケーション** — 組織関係、内部メッセージ、人間への通知方法。
5. **メタ設定** — chat 時の emotion 指示や、特定 engine 向け応答ルール。
6. **現在の状況** — 現在時刻、`current_state.md`、最近の resolution、priming、該当する人間通知、短期記憶。

グループ3には turn で変わる情報がまとまる。先行するグループを安定した prefix として扱うことで、provider の prompt cache を再利用しやすくする。context window が小さい場合は prompt tier に応じて一部の環境情報や optional section を省く。

## スキルと tool guide

スキル一覧は Anima 固有、共通、procedure などを索引化した上で組み立てる。chat では依頼文を skill router に渡して関連候補を優先し、適切な候補がない場合は通常の catalog を使う。catalog の件数は既定で最大3件に抑える。chat では有効化された skill の本文も context に加わる。自動実行では人間の承認が必要なスキルを一覧から除外する。

tool guide は heartbeat、MCP を使う engine、それ以外の engine に応じて切り替わる。スキル本文や tool の実際の入出力は要求に応じて読み込み、system prompt には利用可能な入口と必要な規則を置く。

## budget と肥大化への対処

assembler は section の優先度をもとに予算を配分し、elastic section は Markdown の段落単位で削る。個別段落を途中で切断せず、hard ceiling を超える場合は優先度の低い rigid section も外す。既定の通常 target は 6,000 token、上限は context window の 35% であり、設定によって変更できる。

`AGENTS.md` のサイズ警告は consolidation 中に限り、設定された文字数 threshold を超えた場合に追加される。スキル catalog にも件数上限がある。全体・個別設定の一覧は[設定リファレンス](../reference/config.md)を参照する。

## template の解決

Markdown template は `templates/{locale}/` 以下に置く。`core/paths.py` の `resolve_template_path` は指定 locale、`en`、`ja`、`_shared` の順に探索する。配布形態に応じた runtime template path は、この探索後の互換 fallback として扱われる。
