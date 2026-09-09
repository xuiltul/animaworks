## 런타임 데이터 디렉토리

모든 런타임 데이터는 `{data_dir}/`에 저장되어 있습니다.

```
{data_dir}/
├── company/          # 회사 비전 및 정책 (읽기 전용)
├── animas/          # 모든 Anima 데이터
│   ├── {anima_name}/    # ← 당신
│   └── ...               # 다른 Anima
├── prompts/          # 프롬프트 템플릿 (캐릭터 설계 가이드 등)
├── vault.json        # 공유 크레덴셜 볼트
├── shared/           # Anima 간 공유 영역
│   ├── channels/     # Board 채널 (general.jsonl, ops.jsonl 등)
│   ├── credentials.json  # 레거시 호환 fallback
│   ├── inbox/        # 메시지 inbox
│   └── users/        # 공유 사용자 메모리 (사용자별 하위 디렉토리)
├── common_skills/    # 공유 스킬 (읽기 전용)
└── tmp/              # 작업 디렉토리
    └── attachments/  # 메시지 첨부 파일
```

## 접근 규칙

1. **자신의 디렉토리** (`{data_dir}/animas/{anima_name}/`): 자유롭게 읽기/쓰기 가능
2. **공유 영역** (`{data_dir}/shared/`): 읽기/쓰기 가능. 메시지 전송 및 공유 사용자 메모리에 사용
3. **공용 스킬** (`{data_dir}/common_skills/`): 최상위 멤버(supervisor 미설정)만 쓰기 가능. 나머지는 읽기 전용. 모든 멤버가 사용 가능한 스킬
4. **회사 정보** (`{data_dir}/company/`): 최상위 멤버만 쓰기 가능
5. **프롬프트** (`{data_dir}/prompts/`): 읽기 전용. 캐릭터 설계 가이드 등의 템플릿
6. **다른 Anima의 디렉토리**: permissions.json에 명시된 범위에서만 접근 가능
7. **하위 직원의 디렉토리** (supervisor 전용 — 자식, 손자, 증손자 등 모든 하위에 동일 권한):
   - **관리 파일**: `injection.md`, `cron.md`, `heartbeat.md`, `status.json`은 **읽기/쓰기 가능** (조직 역할 배정 및 설정 변경용)
   - **상태 참조**: `activity_log/`와 `state/current_state.md`는 **읽기 전용**입니다. 부하의 태스크는 권한이 있는 태스크 도구로 확인하세요. 정본 저장소는 호스트 소유이므로 직접 수정하지 마세요.
   - **identity.md**: **읽기 전용** (쓰기 보호)
8. **동료의 activity_log**: 같은 supervisor를 가진 동료의 `activity_log/`는 읽기 가능 (검증용). 쓰기는 불가
