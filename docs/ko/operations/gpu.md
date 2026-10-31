<!-- 自動翻訳ファイル・編集禁止 (AUTO-TRANSLATED, DO NOT EDIT). 正本: docs/ja/operations/gpu.md -->
<!-- i18n: source-sha256=ad37c7808c5ec6e92a59c5c365fcf8be49b9103acc62e9cf615b44a98e45bac3 generated=2026-10-31 engine=luna model=gpt-6-luna-2026-09-22 translator=2 -->

> 확인된 커밋: 581e20f1

# GPU 운영

AnimaWorks는 로컬 embedding과 reranker에서 GPU를 사용할 수 있다. 실행 디바이스 선택과 장애 상태는 `core/infra/gpu.py`이 처리하며, 검색 컴포넌트는 CUDA를 사용할 수 없거나 런타임에 장애가 발생한 경우 CPU로 전환한다.

## 디바이스 설정

`core/config/schemas.py`의 `GPUConfig`는 embedding과 reranker의 디바이스 희망 사항, embedding의 batch size, bulk 처리가 interactive 처리에 양보하는 배치 수를 정의한다. 디바이스는 `auto`, `cuda`, `cpu` 중에서 선택한다. 항목과 기본값은 [설정 참조](../reference/config.md)를 참조한다.

embedding의 `auto`은 기존의 `rag.use_gpu` 설정도 참조하며, 유효하고 안전하게 CUDA를 사용할 수 있는 경우 GPU를 선택한다. reranker의 기본 디바이스는 CPU이다. `cuda`을 명시해도 CUDA 초기화에 실패하면 CPU를 사용한다.

```sh
animaworks config get gpu.embedding_device
animaworks config get gpu.reranker_device
animaworks config set gpu.embedding_device auto
```

## 장애 시 동작

embedding model의 CUDA 로드 또는 encode 중에 GPU 장애가 감지되면 장애 상태를 기록하고, CPU의 모델로 처리를 계속한다. reranker도 CUDA 로드·추론 오류 시 CPU에서 재시도한다. 상태는 health/status 응답의 GPU 정보로 확인할 수 있다. CPU 전환은 처리 지속 방안이며, 드라이버나 GPU 본체의 장애를 수리하는 것은 아니다.

재발하는 경우 서버 로그와 health/status의 오류를 확인한다. 호스트 측에서는 `nvidia-smi`의 인식 상태, 커널·드라이버 기록, 온도, 전원, 연결을 관리자가 조사한다. 장애 원인의 해소와 안전한 재시작을 확인하기 전까지는 GPU를 필수로 하는 작업을 전제로 하지 않는다.

## 호스트 측 장애

NVIDIA Xid 79는 GPU가 PCIe bus에서 분리되었음을 나타낸다. 높은 부하나 전원·연결·열·driver 문제가 원인이 될 수 있으며, 애플리케이션이 CPU로 전환되어도 호스트 장애가 해소되지는 않는다. Xid 79 이후에 Xid 154 등이 이어지면 GPU를 사용 불가로 취급하고, driver stack 복구 또는 호스트 재시작 후 상태를 확인한다.

`nvidia-smi -q -d POWER`으로 전력 설정을 확인하고, `journalctl -k --since "YYYY-MM-DD HH:MM"`로 장애 시각 부근의 kernel 기록을 조사한다. 재발하는 경우 전원 용량과 케이블, PCIe 연결, 냉각, driver 안정성을 관리자가 확인한다. 리포지토리에는 `scripts/gpu-power-guard.sh`와 systemd unit이 있지만, power limit은 GPU와 호스트에 맞게 검증된 값만 관리자가 적용한다. 특정 카드용 예시를 다른 GPU에 그대로 적용하지 않는다.

## 성능 조정

embedding의 batch size를 줄이면 피크 메모리를 억제하기 쉬우나 처리 시간이 길어진다. 대규모 인덱스 업데이트가 대화 처리를 방해하는 경우 bulk yield batches를 줄여 대화 요청에 처리 기회를 양보한다. 설정 변경 후에는 검색 품질과 처리 시간을 실제 데이터가 아닌 적절한 운영 지표로 확인하고, GPU 사용 현황과 함께 조정한다.
