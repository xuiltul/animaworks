# エンジニア専門ガイドライン

## コーディング原則

### 最小変更の原則
- 既存コードへの変更は必要最小限にとどめる
- 大規模リファクタリングは明示的に指示された場合のみ実施
- 「ついでに直す」は禁止。スコープ外の修正は別タスクとして記録する

### オーバーエンジニアリングの回避
- YAGNI（You Aren't Gonna Need It）を徹底する
- 将来の拡張を「予測」してコードを複雑にしない
- 抽象化は3回同じパターンが出てから検討する（Rule of Three）
- シンプルな実装で要件を満たせるなら、それが最善

### セキュリティ（OWASP Top 10 意識）
- ユーザー入力は常にバリデーションする
- SQLクエリにはパラメータバインディングを使う（文字列結合禁止）
- シークレット（API キー、パスワード）をコードにハードコードしない
- ファイルパスの組み立てには `pathlib.Path` を使い、パストラバーサルを防ぐ
- サブプロセス呼び出しには `shell=True` を避け、引数リストを使う

```python
# BAD
subprocess.run(f"ls {user_input}", shell=True)

# GOOD
subprocess.run(["ls", user_input], shell=False)
```

## ツール使用ルール

### ファイル操作は専用ツールを優先
- ファイル読み取り: `Read` を使う（`cat`, `head`, `tail` ではなく）
- ファイル編集: `Edit` を使う（`sed`, `awk` ではなく）
- ファイル書き込み: `Write` を使う（`echo >`, `cat <<EOF` ではなく）
- ファイル検索: `Glob` を使う（`find`, `ls` ではなく）
- 内容検索: `Grep` を使う（`grep`, `rg` ではなく）
- `Bash` は Git 操作、パッケージ管理、ビルド、テスト実行など専用ツールで代替できない場合に使う

### ファイル操作のベストプラクティス
- 既存ファイルの編集を優先し、不要なファイル作成を避ける
- 新規ファイルを作る前に、既存の適切なファイルがないか確認する
- ドキュメントファイル（*.md, README）は明示的に指示された場合のみ作成

## コード品質基準

### 型ヒント必須
```python
from __future__ import annotations

def process_item(name: str, count: int = 0) -> dict[str, int]:
    ...
```

- `str | None` 形式を使用（`Optional[str]` ではなく）
- 関数の引数と戻り値には必ず型ヒントを付ける
- 複雑な型は `TypeAlias` で名前を付ける

### パス操作
```python
from pathlib import Path

# BAD
import os
path = os.path.join(base_dir, "subdir", "file.txt")

# GOOD
path = Path(base_dir) / "subdir" / "file.txt"
```

### docstring（Google スタイル）
```python
def calculate_score(items: list[Item], weight: float = 1.0) -> float:
    """スコアを計算する。

    Args:
        items: 評価対象のアイテムリスト。
        weight: スコアの重み係数。

    Returns:
        計算されたスコア値。

    Raises:
        ValueError: items が空の場合。
    """
```

### ログ設定
```python
import logging
logger = logging.getLogger(__name__)

# print() ではなく logger を使う
logger.info("Processing %d items", len(items))
```

### データモデル
- データ構造の定義には Pydantic Model または dataclass を使う
- 辞書の直接操作よりも構造化されたモデルを優先する

## コミット規約

セマンティックコミット形式を使用:
- `feat:` — 新機能追加
- `fix:` — バグ修正
- `refactor:` — リファクタリング（機能変更なし）
- `docs:` — ドキュメントのみの変更
- `test:` — テストの追加・修正
- `chore:` — ビルド設定、依存関係などの雑務

```
feat: ユーザー認証にOAuth2フローを追加
fix: セッションタイムアウト時のメモリリークを修正
refactor: データベース接続プールをシングルトンに統合
```

## テストガイドライン

- コードを変更したら、関連するテストが通ることを確認する
- 新しい関数・メソッドにはユニットテストを書く
- テストは `tests/` ディレクトリに配置し、対象モジュールと同じ構造を保つ
- テスト実行: `pytest` を使用し、変更に関連するテストを指定して実行する

```bash
# 特定のテストファイルを実行
pytest tests/test_target_module.py -v

# 変更に関連するテストのみ
pytest tests/test_target_module.py::TestClassName::test_method -v
```

## エラーハンドリング

- 裸の `except:` や `except Exception:` を避け、具体的な例外をキャッチする
- エラーメッセージには問題の特定に必要な情報を含める
- リトライロジックには指数バックオフを使う

```python
# BAD
try:
    result = api_call()
except:
    pass

# GOOD
try:
    result = api_call()
except ConnectionError as e:
    logger.warning("API接続失敗 (attempt %d/%d): %s", attempt, max_retries, e)
    raise
```

## 非同期処理

- `async/await` を使い、ブロッキング呼び出しを避ける
- 共有状態には `asyncio.Lock()` を使う
- 長時間のCPUバウンド処理は `asyncio.to_thread()` で逃がす
