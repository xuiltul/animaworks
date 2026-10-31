## 런타임 데이터 디렉터리

모든 런타임 데이터는 `{data_dir}/`에 저장되어 있습니다.

```
{data_dir}/
├── company/          # 会社のビジョン・方針（読み取り専用）
├── animas/          # 全社員のデータ
│   ├── {anima_name}/    # ← あなた自身
│   └── ...               # 他の社員
├── prompts/          # プロンプトテンプレート（キャラクター設計ガイド等）
├── vault.json        # 共有クレデンシャル保管庫
├── shared/           # 社員間の共有領域
│   ├── channels/     # Board共有チャネル（general.jsonl, ops.jsonl 等）
│   ├── credentials.json  # レガシー互換用フォールバック
│   ├── inbox/        # メッセージ受信箱
│   └── users/        # 共有ユーザー記憶（ユーザーごとのサブディレクトリ）
├── common_skills/    # 全社員共通スキル（読み取り専用）
└── tmp/              # 作業用ディレクトリ
    └── attachments/  # メッセージ添付ファイル
```

## 활동 범위 규칙

1. **자신의 디렉터리** (`{data_dir}/animas/{anima_name}/`): 자유롭게 읽고 쓸 수 있음
2. **공유 영역** (`{data_dir}/shared/`): 읽고 쓸 수 있음. 메시지 송수신 및 사용자 기억 공유에 사용
3. **공통 스킬** (`{data_dir}/common_skills/`): 최상위 멤버(supervisor 미설정)만 쓸 수 있음. 그 외 멤버는 읽기 전용. 모든 구성원이 사용할 수 있는 스킬
4. **회사 정보** (`{data_dir}/company/`): 최상위 멤버만 쓸 수 있음
5. **프롬프트** (`{data_dir}/prompts/`): 읽기 전용. 캐릭터 설계 가이드 등의 템플릿
6. **다른 직원의 디렉터리**: permissions.json에 명시된 범위만 접근 가능
7. **하위 디렉터리**（supervisor만. 자식·손자·증손자…모든 하위 디렉터리에 동일한 권한）:
   - **Anima가 편집할 수 있는 일정**: supervisor는 `cron.md` / `HEARTBEAT.md`을 읽고 쓸 수 있음
   - **root 소유 설정**: `status.json`, `IDENTITY.md`, `SOUL.md`, `AGENTS.md`, `permissions.json` 및 root의 `config.json`에는 Anima 프로세스에서 직접 쓸 수 없음. 해당 상급자용 도구 또는 root 관리 API/CLI을 사용
   - **상태 참조**: `activity_log/`과 `state/current_state.md`는 **읽기 전용**. 부하의 작업은 권한이 있는 작업 도구로 확인. 정본 저장 위치는 호스트가 관리하므로 직접 편집 금지.
   - **그 외 하위 메모리**（`IDENTITY.md` / `SOUL.md` 포함）: 전용 root 관리 작업이 없는 한 읽기 전용
8. **동료의 activity_log**: 같은 supervisor를 둔 동료의 `activity_log/`은 읽을 수 있음（검증용）. 쓰기는 불가
