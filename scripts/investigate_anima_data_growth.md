# AnimaWorks データディレクトリ肥大化パターン調査レポート（更新）

調査日時: 2026-03-12 午後

## 1. 調査対象サマリ

- アニマ数: 23（本番）
- データルート: `~/.animaworks/animas/{name}/`
- 全体サイズ: animas 4.9GB, shared 154MB, vectordb 125MB

---

## 2. 各項目の調査結果

### 2.1 assets_backup_* — 重大

| アニマ | バックアップ数 | 合計サイズ |
|--------|---------------|-----------|
| sumire | 21 | 290MB |
| sakura | 16 | 272MB |
| ritsu | 25 | 210MB |
| natsume | 3 | 232MB |
| kotoha | 1 | 114MB |
| shizuku | 1 | 111MB |
| yuki | 1 | 120MB |
| rin | 1 | 104MB |
| sora | 3 | 105MB |
| tsumugi | 1 | 91MB |
| hinata | 2 | 106MB |
| kaede | 9 | 69MB |
| mio | 1 | 55MB |
| mikoto | 1 | 19MB |

**合計**: 86ディレクトリ、**約1.9GB**（前回と同程度）

### 2.2 prompt_logs/ — 要監視

| アニマ | サイズ |
|--------|--------|
| mei | 77M |
| hina | 51M |
| yuki | 36M |
| rin | 27M |
| sakura | 26M |
| ritsu | 20M |
| sumire | 19M |
| mikoto | 18M |
| touka | 17M |
| mio | 14M |
| shizuku, natsume, ayame | 13M |
| tsumugi, kotoha | 11M |
| sanae, runa | 9.4-9.6M |
| hinata | 6.8M |
| shino | 6.1M |
| kaede | 5.2M |
| fuji | 3.8M |
| nagi | 516K |
| sora | 4.0K |

### 2.3 shortterm/ — 要監視

| アニマ | ファイル数 | サイズ |
|--------|-----------|--------|
| sanae | 141 | **9.1M** |
| mikoto | 122 | 1.4M |
| sakura | 113 | 2.8M |
| kotoha | 110 | 2.1M |
| hinata | 107 | 1.4M |
| mei | 105 | 1.2M |
| natsume | 104 | 1.5M |
| mio | 101 | 2.9M |
| rin, ritsu, yuki | 101 | 1.8-2.1M |
| touka | 102 | 1.7M |
| kaede | 102 | 2.5M |
| ayame | 100 | 928K |
| shizuku | 101 | **4.8M** |
| sora | 101 | 4.0M |
| sumire | 87 | 1.3M |
| tsumugi | 51 | 2.5M |
| shino | 43 | 588K |
| runa | 40 | 736K |
| fuji, hina, nagi | 0 | 12K |

sanae の shortterm は 9.1MB で依然突出。

### 2.4 episodes/

| アニマ | ファイル数 | サイズ | 最大ファイル |
|--------|-----------|--------|-------------|
| sakura | 35 | 2.0M | 296K |
| rin | 49 | 1.8M | 204K |
| mio | 41 | 1.6M | 464K |
| yuki | 45 | 1.3M | 364K |
| shizuku | 28 | 860K | 228K |
| natsume | 28 | 848K | 120K |
| kotoha | 44 | 772K | 108K |
| hinata | 27 | 648K | 92K |
| sumire | 26 | 604K | 116K |
| touka | 12 | 576K | 80K |
| sanae | 14 | 472K | 96K |
| runa | 12 | 468K | 116K |
| その他 | 2-30 | 28K-404K | — |

### 2.5 knowledge/

| アニマ | ファイル数 | サイズ |
|--------|-----------|--------|
| ayame | 242 | **7.3M** |
| ritsu | 241 | **6.0M** |
| touka | 114 | 748K |
| sanae | 76 | 1.8M |
| sakura | 70 | 404K |
| shino | 66 | 416K |
| rin | 61 | 392K |
| mikoto | 56 | 260K |
| その他 | 0-45 | 4K-236K |

### 2.6 activity_log/ — 要監視

| アニマ | ファイル数 | 合計 | 最大日次ファイル |
|--------|-----------|------|-----------------|
| sakura | 25 | 87M | 14M (2026-03-04) |
| sanae | 12 | 57M | **15M** (2026-03-08) |
| rin | 24 | 71M | 8.6M (2026-03-02) |
| mikoto | 16 | 71M | 7.1M (2026-03-01) |
| natsume | 23 | 56M | 6.9M (2026-03-07) |
| ayame | 9 | 46M | 12M (2026-03-07) |
| kotoha | 24 | 44M | 4.8M |
| mio | 24 | 42M | 4.4M |
| touka | 12 | 40M | 4.8M |
| sumire | 24 | 38M | 5.2M |
| mei | 15 | 29M | 4.7M |
| runa | 12 | 27M | 5.3M |
| shino | 12 | 24M | 3.0M |
| kaede | 8 | 20M | 4.7M |
| shizuku | 23 | 15M | 1.6M |
| hina | 12 | 8.5M | 2.5M |
| hinata | 21 | 49M | 8.1M |

**1MB超の日次ファイル**: 227件（前回225件→微増）
**最大**: sanae 2026-03-08.jsonl (15MB)、sakura 2026-03-04 (14MB)

### 2.7 transcripts/

| アニマ | ファイル数 | サイズ |
|--------|-----------|--------|
| mei | 11 | 836K |
| mikoto | 11 | 992K |
| sakura | 13 | 792K |
| ritsu | 9 | 436K |
| kotoha | 9 | 152K |
| hina | 2 | 604K |
| その他 | 0-10 | 8K-280K |

### 2.8 shared/channels/

| ファイル | サイズ |
|----------|--------|
| general.jsonl | 304K |
| board.jsonl | 120K |
| ops.jsonl | 216K |
| board.meta.json | 4K |
| **合計** | **648K** |

### 2.9 shared/dm_logs/

- 48ファイル、合計 **540K**
- アーカイブ済み（.archive.jsonl）が主体、現行は0B

### 2.10 各アニマ全体サイズランキング

| 順位 | アニマ | 合計サイズ |
|------|--------|-----------|
| 1 | yuki | 519M |
| 2 | natsume | 486M |
| 3 | sakura | 464M |
| 4 | ritsu | 429M |
| 5 | sumire | 395M |
| 6 | rin | 281M |
| 7 | kaede | 234M |
| 8 | kotoha | 226M |
| 9 | mio | 226M |
| 10 | hinata | 214M |
| 11 | shizuku | 191M |
| 12 | ayame | 168M |
| 13 | mikoto | 161M |
| 14 | mei | 161M |
| 15 | sora | 156M |
| 16 | tsumugi | 151M |
| 17 | sanae | 135M |
| 18 | touka | 107M |
| 19 | hina | 94M |
| 20 | runa | 83M |
| 21 | shino | 71M |
| 22 | fuji | 23M |
| 23 | nagi | 12M |

---

## 3. vectordb（RAG）サイズ

| アニマ | サイズ |
|--------|--------|
| ritsu | 111M |
| ayame | 88M |
| rin | 50M |
| sakura | 48M |
| mio | 45M |
| sanae | 39M |
| natsume | 33M |
| yuki, touka, kotoha, hinata | 31M |
| その他 | 4M-27M |

---

## 4. 前回（2026-03-12午前）との比較

### 改善している点
- **特になし**（同一日の午前→午後のため大きな変化は想定通り少ない）

### 悪化・継続している点

1. **assets_backup**: 86個・約1.9GBで前回と同程度。クリーンアップ未実施
2. **state/pending 滞留**: 依然として同一ファイルが滞留
   - rin: e2e-test-proposal-result.md (12.5日)、prompt_search_results.md, karte_search_results.md (10.3-10.4日)
   - yuki: last_token_rotation_report.md (10.6日)、token_rotator_invalid_grant_error_handling.md (9.3日)
   - mei: aichi-bank-followup.md, ritsu-fudosan-correction.md (9.6日)
3. **activity_log**: 1MB超ファイル 227件（前回225件→+2）
4. **sanae shortterm**: 9.1MB（前回8.8MB→+0.3MB 増加）
5. **prompt_logs**: mei 77M, hina 51M で上位維持。デバッグログのローテーション未実施

### 新たな懸念
- **sanae task_results**: 337ファイル・419.7K、pending failed 9件
- **rin task_queue**: 220KB・115タスク（p:33, o:79）
- **sakura task_queue**: 145KB・101タスク（p:36, o:65）

---

## 5. 推奨アクション（前回から更新）

1. **assets_backup のクリーンアップ**: 30日以上前のバックアップ削除、または最新1件のみ保持
2. **state/pending 滞留の解消**: TaskExec の処理失敗・スキップ原因の調査。古い .wake の削除
3. **activity_log のローテーション**: 90日以上前の日次ファイルをアーカイブまたは削除
4. **shortterm の整理**: sanae の月次サマリ等、不要ファイルの移出
5. **prompt_logs のローテーション**: デバッグ用ログの自動削除（例: 7日保持）
6. **task_queue / task_results の監視**: rin, sakura, sanae のタスク蓄積傾向の調査
