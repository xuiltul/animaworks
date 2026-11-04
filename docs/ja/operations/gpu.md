> 確認したコミット: 581e20f1

# GPU の運用

AnimaWorks はローカルの embedding と reranker で GPU（NVIDIA CUDA または Apple Silicon の MPS）を利用できる。実行デバイスの選択と障害状態は `core/infra/gpu.py` が扱い、検索コンポーネントは GPU が利用できない場合や実行時に障害が起きた場合に CPU へ切り替える。

## デバイスの設定

`core/config/schemas.py` の `GPUConfig` は、embedding と reranker のデバイス希望、embedding の batch size、bulk 処理が interactive 処理へ譲るバッチ数を定義する。デバイスは `auto`、`cuda`、`mps`、`cpu` から選ぶ。項目と既定値は[設定リファレンス](../reference/config.md)を参照する。

`auto` は CUDA、MPS、CPU の順に使えるものを選ぶ。embedding の `auto` は従来の `rag.use_gpu` 設定も参照し、無効なら CPU を使う。reranker の既定デバイスは CPU である。`cuda` や `mps` を明示してもそのデバイスが使えない場合は、使える GPU へ置き換え、なければ CPU を使用する。置き換えはコンポーネントごとに一度だけログへ記録する。

```sh
animaworks config get gpu.embedding_device
animaworks config get gpu.reranker_device
animaworks config set gpu.embedding_device auto
```

## 障害時の動作

embedding model の GPU への読込または encode 中に GPU 障害（MPS のメモリ不足を含む）が検出されると、障害状態を記録し、CPU 上のモデルで処理を続ける。reranker も GPU の読込・推論エラー時には CPU で再試行する。切り替え後はサーバーを再起動するまで CPU のままである。状態は health/status 応答の GPU 情報で確認できる。CPU への切替は処理継続策であり、ドライバーや GPU 本体の障害を修復するものではない。

再発する場合は、サーバーログと health/status のエラーを確認する。CUDA のホスト側では `nvidia-smi` の認識状態、カーネル・ドライバー記録、温度、電源、接続を管理者が調査する。障害原因の解消と安全な再起動を確認するまでは、GPU を必須とする作業を前提にしない。

## ホスト側の障害

NVIDIA Xid 79 は GPU が PCIe bus から切り離されたことを示す。高負荷や電源・接続・熱・driver の問題が原因となる場合があり、アプリケーションが CPU へ切り替わってもホスト障害の解消にはならない。Xid 79 の後に Xid 154 などが続く場合は GPU を利用不可として扱い、driver stack の復旧またはホスト再起動後に状態を確認する。

`nvidia-smi -q -d POWER` で電力設定を確認し、`journalctl -k --since "YYYY-MM-DD HH:MM"` で障害時刻付近の kernel 記録を調べる。再発する場合は、電源容量とケーブル、PCIe 接続、冷却、driver の安定性を管理者が確認する。リポジトリには `scripts/gpu-power-guard.sh` と systemd unit があるが、power limit は GPU とホストに合わせて検証した値だけを管理者が適用する。特定カード向けの例を他の GPU にそのまま適用しない。

## 性能調整

embedding の batch size を小さくするとピークメモリを抑えやすいが、処理時間が長くなる。大規模な索引更新が対話処理を妨げる場合は、bulk yield batches を小さくして対話要求へ処理機会を譲る。設定変更後は、検索品質と処理時間を実データではなく適切な運用指標で確認し、GPU 使用状況と併せて調整する。

## スレッド数

torch のモデルは読込も推論も `core/infra/inference.py` の専用スレッドで実行する。OpenMP は torch を呼んだスレッドごとにワーカー群を作るため、呼び出し元のスレッドで実行するとスレッド数が呼び出し元の数に比例して増える。CPU と CUDA では embedding と reranker が別のスレッドで動き、MPS では両者が同じ一本のスレッドを共有する。torch の MPS は複数スレッドからの同時利用に対応しておらず、同時に呼ぶと Python で捕捉できない Metal のアサーションでプロセスが異常終了するためである。モデルを呼ぶ新しいコードは必ず `run_inference` を経由する。

torch の CPU スレッド数は `gpu.torch_num_threads` と `gpu.torch_num_interop_threads` で上限を決める。ChromaDB 内部の tokio と rayon のスレッド数は `rag.chroma_native_threads` で上限を決める。いずれも `0` でライブラリの既定（CPU コア数）に戻り、変更の反映にはプロセスの再起動が必要である。`TOKIO_WORKER_THREADS` と `RAYON_NUM_THREADS` を環境変数で指定している場合は、その値を優先する。

## MPS での運用

MPS と CPU の embedding は実質的に同一で、既存のベクトルインデックスを再構築する必要はない。`scripts/verify_mps_inference.py` は実モデルで出力の一致、同時実行、メモリ、CPU への切り替えを確認するスクリプトで、CI では実行しない。

`gpu.embedding_batch_size` と `rag.embedding_max_seq_length` が MPS のメモリ使用量を決める。2048 トークンの文書 8 件のバッチでは、モデル読込後の約 2.1 GiB に対してピークが約 3.1 GiB、32 件では約 7.4 GiB だった。解放したバッファは torch が再利用するためバッチごとにキャッシュを消去する必要はなく、メモリ不足になった場合は CPU へ切り替わる。メモリの少ない Mac では batch size を小さくする。
