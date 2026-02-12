from __future__ import annotations

from pathlib import Path

from core.memory import MemoryManager

BEHAVIOR_RULES = """\
## 行動ルール

### 記憶の検索（最重要）
**記憶を検索せずに判断するのは禁止。必ず書庫を引いてから判断すること。**
**ただし検索は内部動作であり、相手に「記憶を検索します」等と実況しないこと。**

判断の前に必ず以下のパターンで記憶を検索してください:
1. 相手の名前やトピックで knowledge/ を Grep 検索
2. 必要に応じて episodes/ も検索（過去に何があったか）
3. 手順が不明なら procedures/ を Read で確認
4. [IMPORTANT] タグの教訓を重視すること
5. 依頼内容がスキルに該当する場合は skills/ から該当スキルをReadで読み、手順に従う

### 記憶の書き込み
行動の後に必ず記憶を更新してください:
1. episodes/{today}.md に行動ログを追記（フォーマット: ## HH:MM {タイトル}）
2. 新しい学びがあれば knowledge/ に書き込み
3. 重要な教訓には [IMPORTANT] タグを付ける
4. state/current_task.md を更新

### 通信ルール
- テキスト + ファイル参照のみ。内部状態の直接共有は禁止
- 自分の言葉で圧縮・解釈して伝える
- 長い内容はファイルとして置き「ここに置いた」と伝える
"""


def _discover_other_persons(person_dir: Path) -> list[str]:
    """List sibling person directories."""
    persons_root = person_dir.parent
    self_name = person_dir.name
    others = []
    for d in sorted(persons_root.iterdir()):
        if d.is_dir() and d.name != self_name and (d / "identity.md").exists():
            others.append(d.name)
    return others


def _build_messaging_section(person_dir: Path, other_persons: list[str]) -> str:
    """Build the messaging instructions with resolved paths."""
    self_name = person_dir.name
    main_py = person_dir.parent.parent / "main.py"

    persons_line = ", ".join(other_persons) if other_persons else "(まだ他の社員はいません)"

    return f"""\
## メッセージ送信（社員間通信）

他の社員にメッセージを送ることができます。送信すると相手は即座に通知されます。

**送信可能な相手:** {persons_line}

**Bashで送信する場合:**
```
python {main_py} send {self_name} <宛先> "メッセージ内容"
```

スレッドで返信する場合:
```
python {main_py} send {self_name} <宛先> "返信内容" --reply-to <元メッセージID> --thread-id <スレッドID>
```

- 受信メッセージの `id` と `thread_id` を使って返信を紐付けること
- 相手が忙しい場合でも、メッセージは inbox に保存され、相手が空いたら自動で処理される
- 返答が必要な依頼には「返答をお願いします」と明記すること"""


def build_system_prompt(memory: MemoryManager) -> str:
    """Construct the full system prompt from Markdown files.

    System prompt =
        identity.md (who you are)
        + injection.md (role/philosophy)
        + permissions.md (what you can do)
        + state/current_task.md (what you're doing now)
        + memory directory guide
        + behavior rules (search-before-decide)
        + messaging instructions
    """
    parts: list[str] = []

    company_vision = memory.read_company_vision()
    if company_vision:
        parts.append(company_vision)

    identity = memory.read_identity()
    if identity:
        parts.append(identity)

    injection = memory.read_injection()
    if injection:
        parts.append(injection)

    permissions = memory.read_permissions()
    if permissions:
        parts.append(permissions)

    state = memory.read_current_state()
    if state:
        parts.append(f"## 現在の状態\n\n{state}")

    pending = memory.read_pending()
    if pending:
        parts.append(f"## 未完了タスク\n\n{pending}")

    # Memory directory guide
    pd = memory.person_dir
    knowledge_list = ", ".join(memory.list_knowledge_files()) or "(なし)"
    episode_list = ", ".join(memory.list_episode_files()[:7]) or "(なし)"
    procedure_list = ", ".join(memory.list_procedure_files()) or "(なし)"
    skill_summaries = memory.list_skill_summaries()
    skill_names = ", ".join(s[0] for s in skill_summaries) or "(なし)"

    parts.append(f"""\
## あなたの記憶（書庫）

全ての記憶は `{pd}/` にあります。

| ディレクトリ | 種類 | 内容 |
|-------------|------|------|
| `{pd}/episodes/` | エピソード記憶 | 過去の行動ログ（日別） |
| `{pd}/knowledge/` | 知識 | 学んだこと・対応方針・ノウハウ |
| `{pd}/procedures/` | 手順書 | 作業の進め方 |
| `{pd}/skills/` | スキル | 実行可能な能力・テンプレート付き手順 |
| `{pd}/state/` | 現在の状態 | 今何をしているか |

知識ファイル: {knowledge_list}
エピソード: {episode_list}
手順書: {procedure_list}
スキル: {skill_names}""")

    if skill_summaries:
        skill_lines = "\n".join(
            f"| {name} | {desc} |" for name, desc in skill_summaries
        )
        parts.append(f"""\
## あなたのスキル

以下のスキルを持っています。使用する際は `{pd}/skills/{{スキル名}}.md` をReadで読んでから実行してください。

| スキル名 | 概要 |
|---------|------|
{skill_lines}""")

    parts.append(BEHAVIOR_RULES)

    # Messaging instructions
    other_persons = _discover_other_persons(pd)
    parts.append(_build_messaging_section(pd, other_persons))

    return "\n\n---\n\n".join(parts)
