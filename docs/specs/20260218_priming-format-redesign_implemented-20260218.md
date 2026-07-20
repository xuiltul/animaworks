# Priming フォーマット再設計 — ASCIIラベル + トピックグルーピング + ポインタ参照

## ステータス

- 作成日: 2026-02-18
- 状態: 実装待ち
- 優先度: 高

## 背景

### 発生した問題

mio が 00:09 に `~/.animaworks/shared/aws_env/` に boto3 用仮想環境を作成。
しかし 05:35 のハートビートで同じ問題について yuki から再度相談された際、
mio は存在しない `/opt/venv/` をハルシネーションして案内した。

### 根本原因

1. **Priming の 200 文字 truncate で具体的なパスが消失**
   - `_format_entry()` (`core/memory/activity.py:252-259`) が `summary or content` を 200 文字で切り詰め
   - 「boto3 問題に対応した」という概要は残るが、作成したパスは消える
2. **LLM が「概要で十分」と誤判断**
   - `behavior_rules.md` (L9-10) は「コンテキスト内の記憶で十分なら応答してよい」と定義
   - 概要レベルの記憶があるため追加検索をスキップ
3. **1 エントリ 1 行の平坦なフォーマット**
   - 同一トピックのやり取りが分散し、対話の文脈が失われる
   - 各行に `from:yuki` 等の冗長な情報が繰り返される
4. **絵文字アイコンのトークン浪費**
   - `📨`, `💓` 等は 2-3 トークン消費するが意味伝達が曖昧

### 設計原則（Claude Code から学んだこと）

- **「行動ログが真実、Priming はヒント（窓）」** — activity_log が source of truth
- **ルール追加ではなくアーキテクチャで解決** — behavior_rules を肥大化させない
- **truncate 時はポインタを残す** — LLM が自分で詳細を読みに行ける構造

## 確定方針

activity_log を source of truth、Priming を「ヒント（窓）」として扱い、
`format_for_priming()` を4段階で再設計する。

## 不採用とした代替案

| 代替案 | 却下理由 |
|--------|---------|
| behavior_rules にルールを大量追加する方式 | プロンプトが肥大化する。Claude Code の設計思想「ルールではなくアーキテクチャで解決」に反する |
| Priming バジェットを単純に増やす方式 | 根本原因はフォーマットの非効率であり、バジェット増加はコンテキスト圧迫を招く |
| 毎回強制的に activity_log を全文読み込む方式 | トークン浪費。Priming はヒントであり、必要時のみ詳細を読む設計が正しい |
| 絵文字をカスタムユニコード文字に置き換え | LLM の認識精度が不確実。ASCII 文字列が最も安定 |

## 変更対象ファイル

| ファイル | 現在の行数 | 変更内容 |
|---------|-----------|---------|
| `core/memory/activity.py` | 293 | `_format_entry()` ASCII化 + ポインタ、`EntryGroup` dataclass 新規、`_group_entries()` / `_format_group()` 新規、`format_for_priming()` グループベース書き換え、`recent()` 行番号付与、`to_dict()` に `_line_number` pop 追加 |
| `core/memory/priming.py` | 428 | heartbeat 時チャネル B バジェット最低保証 |
| `core/prompt/builder.py` | 483 | `_load_recent_activity_summary()` 関数削除、セクション 9 注入コード削除 |
| `~/.animaworks/prompts/behavior_rules.md` | 87（ランタイム） | 「追加検索が必要な典型例」に 1 行追加 |

### テスト影響

| テストファイル | 変更内容 |
|--------------|---------|
| `tests/unit/core/memory/test_activity.py` | `test_format_basic` の絵文字→ASCII アサーション更新、`test_format_long_content_truncated` にポインタ検証追加 |
| `tests/unit/core/memory/test_activity_spec_compliance.py` | `test_memory_write_format_has_icon` の `📝` → `MEM` 、`test_error_format_has_icon` の `❌` → `ERR` |
| `tests/e2e/core/test_spec_compliance_e2e.py` | `test_memory_write_event_recorded` の `\U0001f4dd` → `MEM`、`test_error_events_across_phases` の `\u274c` → `ERR`、`test_cron_command_activity_log_format` の `\u23f0` → `CRON`、`test_from_to_fields_in_priming_pipeline` のフォーマット形式更新 |
| `tests/unit/core/memory/test_activity_grouping.py` | **新規作成** — グルーピング関連テスト 10 件 |
| `tests/unit/core/prompt/test_builder.py` | セクション 9 関連アサーション削除（存在する場合） |
| `tests/test_priming_budget.py` | heartbeat バジェット値の更新（存在する場合） |

## 実装計画

### Phase 1: ASCII ラベル化 + ポインタ参照

後方互換。出力形式の変更のみ。API・JSONL フォーマットに変更なし。

#### 1-1. `ActivityEntry` に `_line_number` フィールド追加

**ファイル**: `core/memory/activity.py:36-49`

```python
@dataclass
class ActivityEntry:
    """A single entry in the unified activity log."""

    ts: str
    type: str
    content: str = ""
    summary: str = ""
    from_person: str = ""
    to_person: str = ""
    channel: str = ""
    tool: str = ""
    via: str = ""
    meta: dict[str, Any] = field(default_factory=dict)
    _line_number: int = field(default=0, init=False, repr=False)
```

`to_dict()` に明示的 pop を追加:

```python
def to_dict(self) -> dict[str, Any]:
    d = asdict(self)
    d.pop("_line_number", None)
    d = {k: v for k, v in d.items() if v}
    # Rename Python field names to JSONL keys
    if "from_person" in d:
        d["from"] = d.pop("from_person")
    if "to_person" in d:
        d["to"] = d.pop("to_person")
    return d
```

#### 1-2. `recent()` で行番号を記録

**ファイル**: `core/memory/activity.py:157-198`

`recent()` 内の JSONL パースループで、`enumerate` を使い行番号を付与:

```python
# 現在のコード（L167）:
for line in path.read_text(encoding="utf-8").splitlines():

# 変更後:
for line_num, line in enumerate(
    path.read_text(encoding="utf-8").splitlines(), start=1
):
```

エントリ生成後に行番号を設定:

```python
entry = ActivityEntry(**{...})
entry._line_number = line_num
entries.append(entry)
```

#### 1-3. `_format_entry()` の type_map を ASCII 化

**ファイル**: `core/memory/activity.py:251-292`

```python
type_map: dict[str, str] = {
    "message_received": "MSG<",
    "response_sent": "MSG>",
    "channel_read": "CH.R",
    "channel_post": "CH.W",
    "dm_received": "DM<",
    "dm_sent": "DM>",
    "human_notify": "NTFY",
    "tool_use": "TOOL",
    "heartbeat_start": "HB",
    "heartbeat_end": "HB",
    "cron_executed": "CRON",
    "memory_write": "MEM",
    "error": "ERR",
}
```

各ラベルは 2-4 文字の ASCII。1 トークンで確実に認識される。
`<` / `>` で方向性（受信/送信）を表現。

#### 1-4. truncate 行末にソースファイルポインタを追加

**ファイル**: `core/memory/activity.py:255-259`

現在:
```python
text = entry.summary or entry.content
if len(text) > 200:
    text = text[:200] + "..."
```

変更後:
```python
text = entry.summary or entry.content
if len(text) > 200:
    date_str = entry.ts[:10] if len(entry.ts) >= 10 else "unknown"
    text = text[:180] + f"...(-> activity_log/{date_str}.jsonl)"
```

- truncate された場合のみポインタを付与
- LLM は「詳細が切れている」と認識し、必要なら `read_memory_file` で読みに行ける
- 180 文字 + ポインタ ≒ 200 文字相当でバジェット影響は最小

#### 1-5. Phase 1 テスト

**既存テスト更新**:

| テストファイル | 変更箇所 |
|--------------|---------|
| `tests/unit/core/memory/test_activity.py` | `test_format_basic`: `message_received` / `response_sent` がフォーマットに含まれる検証はそのまま。追加で `MSG<` / `MSG>` の存在を検証 |
| `tests/unit/core/memory/test_activity.py` | `test_format_long_content_truncated`: `"..."` に加え `"-> activity_log/"` の存在を検証 |
| `tests/unit/core/memory/test_activity_spec_compliance.py` | `test_memory_write_format_has_icon`: `"📝"` → `"MEM"` に変更 |
| `tests/unit/core/memory/test_activity_spec_compliance.py` | `test_error_format_has_icon`: `"❌"` → `"ERR"` に変更 |
| `tests/e2e/core/test_spec_compliance_e2e.py` | `test_memory_write_event_recorded`: `"\U0001f4dd"` → `"MEM"` |
| `tests/e2e/core/test_spec_compliance_e2e.py` | `test_error_events_across_phases`: `"\u274c"` → `"ERR"` |
| `tests/e2e/core/test_spec_compliance_e2e.py` | `test_cron_command_activity_log_format`: `"\u23f0"` → `"CRON"` |

**新規テスト** (`tests/unit/core/memory/test_activity.py` に追加):

- `test_format_entry_uses_ascii_labels` — 全 13 タイプの ASCII ラベル出力を検証
- `test_truncated_entry_has_pointer` — 200 文字超 content でポインタ `(-> activity_log/{date}.jsonl)` が付与されることを検証
- `test_short_entry_no_pointer` — 200 文字以下でポインタなしを検証
- `test_line_number_assigned_by_recent` — `recent()` で読み込んだエントリに `_line_number` が設定されることを検証
- `test_line_number_not_in_jsonl` — `to_dict()` の出力に `_line_number` が含まれないことを検証

#### 1-6. Phase 1 完了条件

- 全 13 タイプの ASCII ラベルが `_format_entry()` で正しく出力される
- 200 文字超 content の truncate 時にポインタ参照が付く
- 200 文字以下ではポインタなし
- `_line_number` が JSONL に漏出しない
- 既存テスト（絵文字→ASCII に更新済み）と新規テスト全件パス

---

### Phase 2: トピックグルーピング

Phase 1 完了が前提（ASCII ラベルをグループヘッダで使用するため）。

#### 2-1. `EntryGroup` dataclass 追加

**ファイル**: `core/memory/activity.py`（`ActivityEntry` の直後に配置）

```python
@dataclass
class EntryGroup:
    """A group of related activity entries for compact priming display."""

    type: str                     # "dm", "hb", "cron", "single"
    start_ts: str                 # グループ開始タイムスタンプ
    end_ts: str                   # グループ終了タイムスタンプ
    entries: list[ActivityEntry]  # グループ内エントリ
    label: str                    # グループラベル（例: "yuki: boto3問題"）
    source_lines: str             # JSONLの行番号範囲（例: "L2-6"）
```

`activity.py` 内部のみで使用。外部 API に影響なし。

#### 2-2. `_group_entries()` メソッド追加

**ファイル**: `core/memory/activity.py`（`ActivityLogger` クラス内、`format_for_priming()` の前に配置）

```python
@staticmethod
def _group_entries(
    entries: list[ActivityEntry],
    time_gap_minutes: int = 30,
) -> list[EntryGroup]:
```

**グルーピングルール（確定）**:

1. **DM グループ**: 同一 peer（`from_person` or `to_person`）の連続する `dm_sent` / `dm_received` を `time_gap_minutes` 以内ならまとめる
2. **HB グループ**: 連続する `heartbeat_start` / `heartbeat_end` のみをグループ化。間に挟まる他タイプのイベント（dm_sent, tool_use 等）は HB グループに含めず、独立したグループとして扱う
3. **CRON グループ**: `meta` 内の同一 `task_name` の `cron_executed` を 1 グループ
4. **それ以外**: `type="single"` の単独グループとして保持

**グルーピングアルゴリズム**:

```python
def _group_entries(entries, time_gap_minutes=30):
    groups = []
    current_group = None
    gap_seconds = time_gap_minutes * 60

    for entry in entries:
        entry_type = entry.type

        # DM判定
        if entry_type in ("dm_sent", "dm_received"):
            peer = entry.to_person if entry_type == "dm_sent" else entry.from_person
            if (current_group
                and current_group.type == "dm"
                and _get_peer(current_group) == peer
                and _time_diff(current_group.end_ts, entry.ts) <= gap_seconds):
                current_group.entries.append(entry)
                current_group.end_ts = entry.ts
                continue
            # 新DMグループ
            if current_group:
                groups.append(current_group)
            current_group = EntryGroup(
                type="dm", start_ts=entry.ts, end_ts=entry.ts,
                entries=[entry], label=_dm_label(peer, entry),
                source_lines="",
            )
            continue

        # HB判定
        if entry_type in ("heartbeat_start", "heartbeat_end"):
            if current_group and current_group.type == "hb":
                current_group.entries.append(entry)
                current_group.end_ts = entry.ts
                continue
            if current_group:
                groups.append(current_group)
            current_group = EntryGroup(
                type="hb", start_ts=entry.ts, end_ts=entry.ts,
                entries=[entry], label="", source_lines="",
            )
            continue

        # CRON判定
        if entry_type == "cron_executed":
            task_name = entry.meta.get("task_name", "")
            if (current_group
                and current_group.type == "cron"
                and _get_task_name(current_group) == task_name):
                current_group.entries.append(entry)
                current_group.end_ts = entry.ts
                continue
            if current_group:
                groups.append(current_group)
            current_group = EntryGroup(
                type="cron", start_ts=entry.ts, end_ts=entry.ts,
                entries=[entry], label=task_name, source_lines="",
            )
            continue

        # その他 → 現グループを閉じて単独エントリ
        if current_group:
            groups.append(current_group)
            current_group = None
        groups.append(EntryGroup(
            type="single", start_ts=entry.ts, end_ts=entry.ts,
            entries=[entry], label="", source_lines="",
        ))

    if current_group:
        groups.append(current_group)

    # source_lines 生成
    for group in groups:
        _set_source_lines(group)

    return groups
```

**ヘルパー関数**:

- `_get_peer(group)`: グループ内最初のエントリから peer 名を取得
- `_dm_label(peer, first_entry)`: `"{peer}: {summary or content[:30]}"` 形式のラベル生成
- `_get_task_name(group)`: グループ内最初のエントリの `meta.task_name` を取得
- `_time_diff(ts1, ts2)`: 2つの ISO タイムスタンプの差を秒で返す
- `_set_source_lines(group)`: グループ内エントリの `_line_number` から `source_lines` を生成

**`source_lines` 生成ロジック**:

```python
def _set_source_lines(group):
    # エントリの日付ごとにグルーピング
    by_date: dict[str, list[int]] = {}
    for e in group.entries:
        if e._line_number > 0:
            date_str = e.ts[:10]
            by_date.setdefault(date_str, []).append(e._line_number)

    parts = []
    for date_str, lines in sorted(by_date.items()):
        lines.sort()
        if lines[-1] - lines[0] == len(lines) - 1:
            # 連続行: L2-6
            ref = f"L{lines[0]}-{lines[-1]}" if len(lines) > 1 else f"L{lines[0]}"
        else:
            # 非連続: L1,3,5
            ref = "L" + ",".join(str(n) for n in lines)
        parts.append(f"activity_log/{date_str}.jsonl#{ref}")

    group.source_lines = " + ".join(parts)
```

日跨ぎグループの場合は `activity_log/2026-02-17.jsonl#L45 + activity_log/2026-02-18.jsonl#L1-3` 形式で複数ファイル参照する。

#### 2-3. `_format_group()` メソッド追加

**ファイル**: `core/memory/activity.py`

```python
@staticmethod
def _format_group(group: EntryGroup) -> str:
```

**出力形式**:

- **DM グループ**:
  ```
  [00:03-00:12] DM yuki: boto3/権限問題
    DM< boto3未インストール、PermissionDeniedエラー
    DM> aws_env venv作成、パス=~/.animaworks/shared/aws_env/
    DM< 了解、確認します
    -> activity_log/2026-02-18.jsonl#L2-6
  ```
  - ヘッダ行: `[HH:MM-HH:MM] DM {label}`
  - 子行: `  {DM<|DM>} {summary or content[:100]}`（最大 100 文字）
  - ポインタ行: `  -> {source_lines}`

- **HB グループ**:
  ```
  [00:03-00:12] HB: inbox確認、yuki対応
    -> activity_log/2026-02-18.jsonl#L1,5
  ```
  - ヘッダ行: `[HH:MM-HH:MM] HB: {heartbeat_end の summary[:50]}`
  - heartbeat_start は省略、heartbeat_end の summary のみ表示
  - ポインタ行: `  -> {source_lines}`

- **CRON グループ**:
  ```
  [06:00] CRON daily-report: exit=0
    -> activity_log/2026-02-18.jsonl#L10
  ```
  - ヘッダ行: `[HH:MM] CRON {task_name}: exit={exit_code}`
  - ポインタ行: `  -> {source_lines}`

- **単独エントリ**:
  ```
  [10:30] MSG< message_received(from:user): Hello
  ```
  - Phase 1 の `_format_entry()` と同等の 1 行出力
  - ポインタなし（truncate 時のみ Phase 1 のポインタが付く）

**時刻表示**: start_ts == end_ts の場合は `[HH:MM]`、異なる場合は `[HH:MM-HH:MM]`。

#### 2-4. `format_for_priming()` の書き換え

**ファイル**: `core/memory/activity.py:211-249`

```python
def format_for_priming(
    self,
    entries: list[ActivityEntry],
    budget_tokens: int = 1300,
) -> str:
    if not entries:
        return ""

    groups = self._group_entries(entries)
    max_chars = budget_tokens * _CHARS_PER_TOKEN
    lines: list[str] = []
    total_chars = 0

    # 新しい順にグループを処理（バジェット超過で古いグループから切り捨て）
    for group in reversed(groups):
        formatted = self._format_group(group)
        if total_chars + len(formatted) + 1 > max_chars:
            break
        lines.append(formatted)
        total_chars += len(formatted) + 1  # +1 for newline

    if not lines:
        return ""

    lines.reverse()
    return "\n".join(lines)
```

**注意**: `_format_entry()` は Phase 1 で ASCII 化済み。Phase 2 では単独エントリの表示で引き続き使用する。

#### 2-5. Phase 2 テスト

**新規テストファイル**: `tests/unit/core/memory/test_activity_grouping.py`

| テスト名 | 検証内容 |
|---------|---------|
| `test_dm_entries_grouped_by_peer` | 同一相手の DM が 1 グループにまとまる |
| `test_dm_entries_split_by_time_gap` | 30 分超間隔で別グループに分割される |
| `test_dm_entries_split_by_different_peer` | 異なる相手の DM が別グループになる |
| `test_heartbeat_grouped` | heartbeat_start + heartbeat_end が 1 グループ |
| `test_heartbeat_excludes_interleaved_events` | HB 間に挟まる dm_sent 等は HB グループに含まれず独立 |
| `test_cron_entries_grouped_by_task_name` | 同一 task_name の cron が 1 グループ |
| `test_mixed_types_grouped` | DM + HB + CRON + 単独の混合が正しくグループ化 |
| `test_group_has_source_lines` | source_lines に JSONL 行番号が含まれる |
| `test_group_cross_day_source_lines` | 日跨ぎグループで複数ファイル参照が生成される |
| `test_format_group_dm` | DM グループの出力形式検証（ヘッダ + 子行 + ポインタ） |
| `test_format_group_hb` | HB グループの出力形式検証（end の summary のみ） |
| `test_format_group_single` | 単独エントリが `_format_entry()` 相当の出力 |
| `test_format_for_priming_budget_cuts_old_groups` | バジェット超過時に古いグループから切り捨て |

**既存テスト更新**:

| テストファイル | 変更内容 |
|--------------|---------|
| `tests/unit/core/memory/test_activity.py` | `TestFormatForPriming` クラスの `test_format_basic` をグループ形式の出力に合わせて更新。`test_format_budget_truncation` はグループ単位切り捨てに更新 |
| `tests/e2e/core/test_activity_log_e2e.py` | Priming 統合テストの出力形式検証をグループ形式に更新（該当テストが存在する場合） |
| `tests/e2e/core/test_spec_compliance_e2e.py` | `test_from_to_fields_in_priming_pipeline` のアサーションをグループ形式に更新 |

#### 2-6. Phase 2 完了条件

- 同一 peer の連続 DM（30 分以内）が 1 グループにまとまる
- 30 分超間隔の DM が別グループに分割される
- heartbeat_start + heartbeat_end のみが HB グループになる（間の他イベントは独立）
- 同一 task_name の cron_executed が 1 グループにまとまる
- グループに source_lines（行番号範囲）が含まれる
- 日跨ぎグループで複数ファイル参照が正しく生成される
- `format_for_priming()` がバジェット超過時に古いグループから切り捨てる
- 全テストパス

---

### Phase 3: セクション 9/10 統合

Phase 2 完了が前提（グループ形式でないとセクション 9 削除の効果が薄い）。

#### 3-1. 問題: 二重注入

`builder.py` のセクション 9 (`_load_recent_activity_summary`, L26-54) と
セクション 10 (`priming_section` 内チャネル B) が同じ activity_log から
異なるフィルタで取得しており、情報が重複する。

- セクション 9: `days=1, limit=5`, 特定タイプのみ（heartbeat_end, cron, dm, channel_post, human_notify）
- チャネル B: `days=2, limit=50`, 全タイプ、スコアリング付き

#### 3-2. `_load_recent_activity_summary()` を削除

**ファイル**: `core/prompt/builder.py`

削除対象:
- `_load_recent_activity_summary()` 関数（L26-54）
- `build_system_prompt()` 内のセクション 9 注入コード（L408-416）:
  ```python
  # 削除:
  recent_summary = _load_recent_activity_summary(memory.anima_dir)
  if recent_summary:
      parts.append(
          "## 直近の活動\n\n"
          "以下は最近の活動です。"
          "対話の文脈として考慮してください。\n\n"
          f"{recent_summary}"
      )
  ```

#### 3-3. heartbeat 時チャネル B バジェット最低保証

**ファイル**: `core/memory/priming.py:164-165`

セクション 9 削除により、heartbeat トリガー時にチャネル B のバジェットが
200 トークン（heartbeat 全体バジェット）× 比率 で小さくなりすぎるリスクがある。

変更:
```python
# 現在（L165）:
budget_activity = int(_BUDGET_RECENT_ACTIVITY * (token_budget / _DEFAULT_MAX_PRIMING_TOKENS))

# 変更後:
budget_activity = max(400, int(_BUDGET_RECENT_ACTIVITY * (token_budget / _DEFAULT_MAX_PRIMING_TOKENS)))
```

これにより heartbeat 時でも最低 400 トークン（≒1600 文字）がチャネル B に確保される。

#### 3-4. Phase 3 テスト

| テストファイル | 変更内容 |
|--------------|---------|
| `tests/unit/core/prompt/test_builder.py` | セクション 9（`直近の活動`）関連のアサーションがあれば削除 |
| `tests/test_priming_budget.py` | heartbeat バジェットの最低値が 400 であることを検証（テストが存在する場合） |

新規テスト（`test_activity.py` or `test_priming.py` に追加）:
- `test_heartbeat_budget_minimum_400` — `_BUDGET_HEARTBEAT=200` 時でも `budget_activity` が 400 以上

#### 3-5. Phase 3 完了条件

- `_load_recent_activity_summary()` が削除されている
- `build_system_prompt()` の出力に「直近の活動」セクションが含まれない
- heartbeat 時のチャネル B バジェットが最低 400 トークン
- 全テストパス

---

### Phase 4: behavior_rules の微修正

Phase 1 完了後いつでも実施可能。

#### 4-1. ポインタ参照の例示追加

**ファイル**: `~/.animaworks/prompts/behavior_rules.md` L20 の後に 1 行追加

現在（L16-20）:
```markdown
#### 追加検索が必要な典型例
- 具体的な日時・数値を正確に答える必要がある時
- 過去の特定のやり取りの詳細を確認したい時
- 手順書（procedures/）に従って作業する時
- コンテキストに該当する記憶がない未知のトピックの時
```

変更後:
```markdown
#### 追加検索が必要な典型例
- 具体的な日時・数値を正確に答える必要がある時
- 過去の特定のやり取りの詳細を確認したい時
- 手順書（procedures/）に従って作業する時
- コンテキストに該当する記憶がない未知のトピックの時
- Priming に `->` ポインタがある場合、具体的なパスやコマンドを回答する必要があるとき
```

これはルール追加ではなく既存ルールの例示補足。Phase 1-2 でポインタ構造が入れば
LLM は構造的に「truncate された」と認識できるため、この行がなくても動作する
可能性が高いが、明示しておく。

#### 4-2. Phase 4 完了条件

- `behavior_rules.md` にポインタ参照の例示が 1 行追加されている
- 既存のセクション構造（見出し・行数）に影響がない

## リスク評価

| リスク | 影響度 | 発生確率 | 緩和策 |
|--------|-------|---------|--------|
| グルーピングで重要エントリが埋もれる | 低 | 低 | 単独エントリ（dm/hb/cron 以外）はグループ化しないため埋もれない |
| ポインタ参照で LLM が毎回ファイルを読む | 低 | 低 | ポインタは truncate 時のみ付与。全文が収まる場合はポインタなし |
| 行番号が JSONL 追記で変わる | なし | なし | 日次ファイルは append-only。既存行の番号は不変 |
| Phase 3 でセクション 9 削除後 heartbeat 時の情報不足 | 中 | 低 | チャネル B バジェット最低 400 トークン保証で対策 |
| `_line_number` が JSONL に漏出 | 中 | なし | `to_dict()` で明示的 `pop("_line_number", None)` |
| 日跨ぎグループのポインタ形式が長すぎる | 低 | 低 | 実際には 30 分以内グルーピングのため日跨ぎは稀 |

## 後方互換性

- **Phase 1**: 出力形式の変更のみ。API・JSONL フォーマットに変更なし。絵文字に依存する外部ツールなし（Web UI は `server/routes/` で独自フォーマット）
- **Phase 2**: `format_for_priming()` の出力形式が変わるが、呼び出し元（`priming.py` チャネル B）は戻り値を文字列として扱うだけ。`EntryGroup` は `activity.py` 内部のみ
- **Phase 3**: `_load_recent_activity_summary()` 削除はシステムプロンプトのセクション構成に影響するが、チャネル B がカバーするため機能欠損なし
- **Phase 4**: ランタイムファイルの微修正。コード変更なし

## 稼働中 Anima への影響

- Priming フォーマットの変更はシステムプロンプト構築時に適用される
- activity_log の JSONL フォーマット自体は変更しない
- サーバー再起動後、次のメッセージ/ハートビートから新フォーマットが適用される
- 既存の activity_log ファイルはそのまま読み込み可能

## 見積り

| Phase | 変更ファイル数 | 新規テスト | 既存テスト更新 |
|-------|-------------|-----------|--------------|
| Phase 1 | 1 (activity.py) | 5 | 3 ファイル |
| Phase 2 | 1 (activity.py) | 13 | 3 ファイル |
| Phase 3 | 2 (builder.py, priming.py) | 1 | 既存あれば更新 |
| Phase 4 | 0 (ランタイムのみ) | 0 | 0 |
