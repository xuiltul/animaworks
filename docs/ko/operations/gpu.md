<!-- 自動翻訳ファイル・編集禁止 (AUTO-TRANSLATED, DO NOT EDIT). 正本: docs/ja/operations/gpu.md -->
<!-- i18n: source-sha256=d1884fcbfff860b73ade714750f2d4ff0ac207538925f825253f91797e19509f generated=2026-11-04 engine=local model=deepseek-v4.1-flash-1017-abl translator=2 -->

> 확인된 커밋: 581e20f1

# GPU 운영

AnimaWorks는 로컬 embedding과 reranker에서 GPU(NVIDIA CUDA 또는 Apple Silicon의 MPS)를 사용할 수 있다. 실행 디바이스 선택과 장애 상태는 `core/infra/gpu.py`가 처리하며, 검색 컴포넌트는 GPU를 사용할 수 없거나 런타임에 장애가 발생하면 CPU로 전환한다.

## 디바이스 설정

`core/config/schemas.py`의 `GPUConfig`는 embedding과 reranker의 디바이스 희망, embedding의 batch size, bulk 처리가 interactive 처리에 양보하는 배치 수를 정의한다. 디바이스는 `auto`, `cuda`, `mps`, `cpu` 중에서 선택한다. 항목과 기본값은 [설정 참조](../reference/config.md)를 참조한다.

`auto`은 CUDA, MPS, CPU 순으로 사용 가능한 것을 선택한다. embedding의 `auto`은 기존의 `rag.use_gpu` 설정도 참조하며, 비활성화되어 있으면 CPU를 사용한다. reranker의 기본 디바이스는 CPU이다. `cuda`이나 `mps`을 명시해도 해당 디바이스를 사용할 수 없는 경우에는 사용 가능한 GPU로 대체하고, 없으면 CPU를 사용한다. 대체는 컴포넌트별로 한 번만 로그에 기록한다.

```sh
animaworks config get gpu.embedding_device
animaworks config get gpu.reranker_device
animaworks config set gpu.embedding_device auto
```

## 장애 시 동작

embedding model의 GPU 로드 또는 encode 중에 GPU 장애(MPS의 메모리 부족 포함)가 감지되면 장애 상태를 기록하고 CPU 상의 모델로 처리를 계속한다. reranker도 GPU 로드·추론 오류 시에는 CPU로 재시도한다. 전환 후에는 서버를 재시작할 때까지 CPU 상태로 유지된다. 상태는 health/status 응답의 GPU 정보에서 확인할 수 있다. CPU로의 전환은 처리 지속 대책이며, 드라이버나 GPU 본체의 장애를 복구하는 것은 아니다.

재발하는 경우에는 서버 로그와 health/status의 오류를 확인한다. CUDA의 호스트 측에서는 `nvidia-smi`의 인식 상태, 커널·드라이버 기록, 온도, 전원, 연결을 관리자가 조사한다. 장애 원인의 해소와 안전한 재시작을 확인할 때까지는 GPU를 필수로 하는 작업을 전제로 하지 않는다.

## 호스트 측 장애

NVIDIA Xid 79는 GPU가 PCIe bus에서 분리되었음을 나타낸다. 높은 부하나 전원·연결·열·driver 문제가 원인이 될 수 있으며, 애플리케이션이 CPU로 전환되어도 호스트 장애가 해소되지는 않는다. Xid 79 이후에 Xid 154 등이 이어지면 GPU를 사용 불가로 취급하고, driver stack 복구 또는 호스트 재시작 후 상태를 확인한다.

`nvidia-smi -q -d POWER`으로 전력 설정을 확인하고, `journalctl -k --since "YYYY-MM-DD HH:MM"`로 장애 시각 부근의 kernel 기록을 조사한다. 재발하는 경우 전원 용량과 케이블, PCIe 연결, 냉각, driver 안정성을 관리자가 확인한다. 리포지토리에는 `scripts/gpu-power-guard.sh`와 systemd unit이 있지만, power limit은 GPU와 호스트에 맞게 검증된 값만 관리자가 적용한다. 특정 카드용 예시를 다른 GPU에 그대로 적용하지 않는다.

## 성능 조정

embedding의 batch size를 줄이면 피크 메모리를 억제하기 쉬우나 처리 시간이 길어진다. 대규모 인덱스 업데이트가 대화 처리를 방해하는 경우 bulk yield batches를 줄여 대화 요청에 처리 기회를 양보한다. 설정 변경 후에는 검색 품질과 처리 시간을 실제 데이터가 아닌 적절한 운영 지표로 확인하고, GPU 사용 현황과 함께 조정한다.

## 스레드 수

torch의 모델은 로드도 추론도 `core/infra/inference.py`의 전용 스레드에서 실행한다. OpenMP는 torch를 호출한 스레드마다 워커 그룹을 만들기 때문에, 호출원의 스레드에서 실행하면 스레드 수가 호출원의 수에 비례하여 증가한다. CPU와 CUDA에서는 embedding과 reranker가 별도의 스레드에서 동작하고, MPS에서는 둘이 같은 하나의 스레드를 공유한다. torch의 MPS는 여러 스레드에서의 동시 이용에 대응하지 않으며, 동시에 호출하면 Python에서 포착할 수 없는 Metal의 assertion으로 프로세스가 비정상 종료되기 때문이다. 모델을 호출하는 새로운 코드는 반드시 `run_inference`을 경유한다.

torch의 CPU 스레드 수는 `gpu.torch_num_threads`와 `gpu.torch_num_interop_threads`으로 상한을 정한다. ChromaDB 내부의 tokio와 rayon의 스레드 수는 `rag.chroma_native_threads`로 상한을 정한다. 모두 `0`로 라이브러리의 기본값(CPU 코어 수)으로 돌아가며, 변경의 반영에는 프로세스의 재시작이 필요하다. `TOKIO_WORKER_THREADS`과 `RAYON_NUM_THREADS`을 환경 변수로 지정하고 있는 경우에는 그 값을 우선한다.

## MPS에서의 운영

MPS와 CPU의 embedding은 실질적으로 동일하며, 기존의 벡터 인덱스를 재구축할 필요는 없다. `scripts/verify_mps_inference.py`은 실모델로 출력의 일치, 동시 실행, 메모리, CPU로의 전환을 확인하는 스크립트이며, CI에서는 실행하지 않는다.

`gpu.embedding_batch_size`과 `rag.embedding_max_seq_length`가 MPS의 메모리 사용량을 결정한다. 2048 토큰의 문서 8건 배치에서는 모델 로드 후 약 2.1 GiB에 대해 피크가 약 3.1 GiB, 32건에서는 약 7.4 GiB였다. 해제한 버퍼는 torch가 재사용하므로 배치마다 캐시를 삭제할 필요는 없으며, 메모리가 부족해지면 CPU로 전환된다. 메모리가 적은 Mac에서는 batch size를 작게 한다.
