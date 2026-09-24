# 캐릭터 설계 가이드

새로운 Digital Anima의 캐릭터 설계(또는 자신의 캐릭터 설정)를 위한 공통 규칙입니다.
최소한의 정보(이름, 역할, 성격 방향)에서 일관성 있고 깊이 있는 캐릭터를 만들어 내세요.

## 생성 규칙

### 이름 설계

- 일본어 이름이 미지정이면, 역할과 이미지에 맞는 성과 이름을 창작하세요
- 한자 + 후리가나를 사용하세요. 성과 이름에 통일된 세계관을 부여하세요
- 영문 이름과 음의 연관이 있으면 좋습니다 (예: 영문 이름 → 한자 이름에 음의 연결)

### 얼굴 타입

- `{data_dir}/prompts/face_types.md`에서 **하나만** 선택하세요
- 쿨 계열은 조직당 최대 2명입니다. 이미 2명이 있으면 선택하지 마세요
- 같은 얼굴 타입이 3명 이상이 되는 채용은 다른 타입으로 다시 배정하세요

### 외형 설계

- 역할과 성격에서 연상되는 외형을 설계하세요
- 헤어스타일, 머리색, 눈 색상은 성격 및 이미지 컬러와 조화시키세요
- 얼굴 타입은 "귀여운 계열", "미인 계열", "쿨 계열", "미스터리어스 계열" 등에서 성격에 맞게 선택하세요
- 키와 체중은 나이에 맞는 자연스러운 범위로 설정하세요

### 성격 설계

- "한 마디로"는 짧은 캐치프레이즈입니다. 역할과 성격의 본질을 한 문장으로 표현하세요
- 성격은 2~3문장으로. 장점과 단점(매력적인 약점)을 포함하세요
- 말투는 구체적인 대사 예시를 3개 이상 작성하세요. 1인칭과 어미 패턴도 명확히 정의하세요
- 취미와 특기: 역할과 성격에서 자연스럽게 도출되는 것을 각 3개씩
- 좋아하는 것/싫어하는 것: 역할에서의 "이상적 상태"와 "스트레스 요인"에서 도출
- 동기부여: 따옴표가 있는 캐치프레이즈 형식으로

### AI 직원으로서의 개성

- 실제 업무에서 어떻게 행동하는지 구체적인 행동 패턴 3~4개
- 마지막에 따옴표가 있는 캐치프레이즈 1개

### 이미지 컬러

- 성격과 역할에서 연상되는 색상을 선택하세요
- 일본어 색상 이름 + HEX 코드 (예: 桜色 (#FFB7C5))

## 내부 일관성 체크

설계 완료 후 다음을 확인하세요:

- 생년월일 → 별자리가 맞는지
- 성격 → 말투 → 취미 → 좋아하는 것/싫어하는 것이 일관되는지
- 역할 → AI 직원으로서의 개성이 자연스럽게 연결되는지
- 이미지 컬러와 머리색, 눈 색상의 전체적인 컬러 밸런스

---

## 아바타 이미지 생성

캐릭터 설계가 완료되면 `image_gen` 도구로 아바타 이미지 세트를 생성하세요.
`image_gen`을 사용할 수 있는 경우(permissions.json에서 `image_gen: yes`)에만 실행하세요.

### NovelAI 프롬프트 변환

identity.md의 외형 설정을 NovelAI 호환 애니메 태그로 변환합니다.

**기본 구조:**

```
masterpiece, best quality, very aesthetic, absurdres, anime coloring, clean lineart, soft shading, 1girl/1boy, {hair_color} hair, {hairstyle}, {eye_color} eyes, {outfit}, full body, standing, white background, looking at viewer
```

**변환 예시:**

| identity.md 외형 | NovelAI 프롬프트 |
|---|---|
| 158cm, black long hair, red eyes, sailor uniform | `masterpiece, best quality, very aesthetic, absurdres, anime coloring, clean lineart, soft shading, 1girl, black hair, long hair, red eyes, sailor uniform, full body, standing, white background, looking at viewer` |
| 175cm, silver short hair, blue eyes, suit | `masterpiece, best quality, very aesthetic, absurdres, anime coloring, clean lineart, soft shading, 1boy, silver hair, short hair, blue eyes, business suit, full body, standing, white background, looking at viewer` |

**품질 및 화풍 태그 (프롬프트 앞에 추가):**

프롬프트 시작 부분에 반드시 다음 품질 및 아트 스타일 태그를 포함하세요.

- 품질: `masterpiece, best quality, very aesthetic, absurdres`
- 화풍: `anime coloring, clean lineart, soft shading`

> 참고: NovelAI의 `qualityToggle` 설정으로 품질 태그가 자동 적용될 수 있지만, 프롬프트에 명시하면 더 안정적인 품질을 얻을 수 있습니다.

**캐릭터 속성 태그:**

- 머리색: `black hair`, `brown hair`, `blonde hair`, `silver hair`, `red hair`, `blue hair`, `pink hair`, `white hair`
- 헤어스타일: `long hair`, `short hair`, `medium hair`, `ponytail`, `twintails`, `bob cut`, `braided hair`
- 눈 색상: `{color} eyes` (보석 비유가 아닌 색상 이름을 사용)
- 의상: 구체적인 아이템 이름 (`school uniform`, `business suit`, `lab coat`, `hoodie`, `maid outfit`)
- 필수 후미 태그: `full body, standing, white background, looking at viewer`

**네거티브 프롬프트 (권장):**

```
lowres, bad anatomy, bad hands, missing fingers, extra digits, fewer digits, worst quality, low quality, blurry, jpeg artifacts, cropped, multiple views, logo, too many watermarks
```

### 생성 절차

시스템 프롬프트의 "외부 도구" 섹션에 문서화된 **image_gen** (`generate_character_assets`) 사용법을 따르세요. `steps` 인수는 지정하지 마세요. 선택한 스타일에 따라 생성 에셋이 결정됩니다.

**realistic 스타일:** 자연어 사진 프롬프트를 사용합니다. 전신 사진, 표정별 버스트업 사진, 아이콘만 생성합니다. 치비, 3D 모델, 리깅, 애니메이션은 생성하지 **않습니다**.

realistic 결과물은 `assets/`에 저장됩니다:
- `avatar_fullbody_realistic.png` — 전신 사진
- `avatar_bustup_realistic.png`, `avatar_bustup_{emotion}_realistic.png` — 버스트업 및 표정 차이
- `icon_realistic.png` — 아이콘

**anime 스타일:** 위 변환 규칙의 애니메 태그를 사용하고 전체 애니메 에셋을 생성합니다:
- `avatar_fullbody.png` — 전신 입상 (NovelAI V4.5)
- `avatar_bustup.png` — 버스트업 (Flux Kontext)
- `avatar_chibi.png` — 치비 캐릭터 (Flux Kontext)
- `avatar_chibi.glb` — 3D 모델 (Meshy Image-to-3D)
- `avatar_chibi_rigged.glb` — 리깅된 3D 모델 (Meshy Rigging)
- `anim_walking.glb`, `anim_running.glb` — 기본 애니메이션
- `anim_idle.glb`, `anim_sitting.glb`, `anim_waving.glb`, `anim_talking.glb` — 추가 애니메이션

인수:
- `prompt`: 위 규칙에 따른 스타일별 프롬프트
- `negative_prompt`: 권장 네거티브 프롬프트
- `anima_dir`: 대상 Anima의 디렉토리 (자기 자신 또는 다른 사람의 것)

특정 단계가 실패하면 에러를 기록하고 성공한 출력만 사용하세요.
