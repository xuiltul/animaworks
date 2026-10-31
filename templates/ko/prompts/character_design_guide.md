# 캐릭터 설계 가이드

새로운 Digital Anima 캐릭터 설계(또는 자신의 캐릭터 설정)를 위한 공통 규칙.
인격이 먼저, 직능은 나중. 역할에서 '진지한 쿨 미녀'를 연상해서는 안 된다.

얼굴 타입 정본: 같은 디렉토리의 `face_types.md`

## 생성 규칙

순서는 **성격 → 얼굴 타입 → 외모 → 말투**. 역할은 마지막에 얹는다.

### 이름 설계

- 일본어 이름이 미지정이면, 역할·이미지에 맞는 성명을 창작한다
- 성명은 한자 + 후리가나. 성과 이름에 통일된 세계관을 갖게 한다
- 영문 이름과의 발음 연관이 있으면 좋다 (예: 영문 이름 → 한자 이름에 음의 연결을 갖게 한다)

### 성격 설계 (먼저 결정)

- 직능이 아니라, 인간으로서의 온도로 결정한다. 경리가 갸루여도, 모니터링이 탐정 놀이여도 좋다
- '한마디로'는 짧은 캐치프레이즈. 역할 이름을 반복하지 않는다
- 성격은 2~3문장. 장점과 단점(애교 있는 약점)을 포함한다
- 말투는 구체적인 대사 예를 3개 이상. 1인칭·어미의 특징도 명확히. 전원이 정중어 기반이 되지 않게 한다
- 취미는 **일의 연장 금지**. 3개 중 2개 이상은 직능과 무관하게
- 특기는 일에 사용해도 좋다
- 좋아함/싫어함은 생활의 취향을 섞는다. 장부가 좋아, 만으로 하지 않는다
- 동기부여는 「」 붙은 결의 대사. 즐거움이 보일 것

### 얼굴 타입

- `{data_dir}/prompts/face_types.md` 에서 **1개만** 선택
- 쿨 계열은 조직당 최대 2명. 이미 2명 있다면 선택하지 않는다
- 같은 얼굴 타입이 3명 이상이 되는 채용은, 다른 타입으로 다시 배정한다

### 외모 설계

- 성격과 얼굴 타입에서 결정한다. **역할에서 연상하지 않는다**
- 머리색은 조직 내에서 중복을 피한다. 검정·남색은 쿨 프레임 외의 기본값으로 하지 않는다
- 기본 표정은 미소 (쿨 계열만 무표정 가능)
- 옷은 사복 위주로 개성을 낸다. 전원 정장은 금지
- 키는 나이에 맞는 자연스러운 범위

### AI 사원으로서의 개성

- 직능 자체는 바꾸지 않는다. 바꿀 수 있는 것은 접근 방식의 온도
- 실제 업무에서 어떻게 움직이는지 구체적인 행동 패턴을 3~4개
- 마지막에 결의 대사를 1개 (「」 붙여서)

### 이미지 컬러

- 성격에서 연상되는 색을 선택한다. 직능의 이미지 컬러(경리=남색)에 끌려가지 않는다
- 일본어 색 이름 + HEX 코드 (예: 벚꽃색 (#FFB7C5))

### IDENTITY.md / SOUL.md 의 정본 서식

`IDENTITY.md`는 프로필과 외모, `SOUL.md`는 인격(성격·말투·가치관·개성)의 정본. character_sheet 이나 prompt 에만 있는 외모는 금지.

- IDENTITY.md 필수 섹션: 기본 프로필 (이름·영문 이름·나이·생일·별자리·혈액형·키·소속·직책·상급자·얼굴 타입), 외모, 이미지 컬러, AI 모델 정보
- SOUL.md 필수 섹션: 성격·캐릭터, 말투·가치관, AI 사원으로서의 개성, 결정적 한마디

## 내부 정합성 체크

설계가 완료되면, 다음을 확인할 것:

- 생일→별자리가 올바른가
- 성격→말투→취미→좋아함/싫어함이 모순되지 않는가
- 취미가 전부 직능의 연장이 되지 않았는가
- 역할→AI 사원으로서의 개성이 자연스럽게 이어지는가 (직능은 유지, 온도는 인격)
- 이미지 컬러와 머리색·눈동자 색의 전체적인 컬러 밸런스
- 기존 멤버와 얼굴 타입·머리색·말투가 겹치지 않는가

---

## 아바타 이미지 생성

캐릭터 설계가 완료되면, `image_gen` 도구로 아바타 이미지 일체를 생성한다.
`image_gen` 이 사용 가능한 경우 (permissions.json 에서 image_gen 이 허가) 에만 실행할 것.

### NovelAI 프롬프트로의 변환

IDENTITY.md 의 외모 설정을 NovelAI 호환 애니메 태그로 변환한다.

**기본 구조:**

```
masterpiece, best quality, very aesthetic, absurdres, anime coloring, clean lineart, soft shading, 1girl/1boy, {hair_color} hair, {hairstyle}, {eye_color} eyes, {outfit}, full body, standing, white background, looking at viewer
```

**변환 예:**

| IDENTITY.md 의 외모 | NovelAI 프롬프트 |
|---|---|
| 키 158cm·검은 머리 롱·빨간 눈·세일러복 | `masterpiece, best quality, very aesthetic, absurdres, anime coloring, clean lineart, soft shading, 1girl, black hair, long hair, red eyes, sailor uniform, full body, standing, white background, looking at viewer` |
| 키 175cm·은발 숏·파란 눈·정장 | `masterpiece, best quality, very aesthetic, absurdres, anime coloring, clean lineart, soft shading, 1boy, silver hair, short hair, blue eyes, business suit, full body, standing, white background, looking at viewer` |

**품질·화풍 태그 (앞에 부여):**

프롬프트 앞에 다음 품질 태그와 아트 스타일 태그를 반드시 포함할 것.

- 품질: `masterpiece, best quality, very aesthetic, absurdres`
- 화풍: `anime coloring, clean lineart, soft shading`

> 주의: NovelAI 의 `qualityToggle` 설정에서도 품질 태그가 자동 부여되지만, 프롬프트에 명시하면 더 안정적인 품질을 얻을 수 있다.

**캐릭터 속성 태그:**

- 머리색: `black hair`, `brown hair`, `blonde hair`, `silver hair`, `red hair`, `blue hair`, `pink hair`, `white hair`
- 헤어스타일: `long hair`, `short hair`, `medium hair`, `ponytail`, `twintails`, `bob cut`, `braided hair`
- 눈동자 색: `{color} eyes` (보석의 비유가 아니라 색 이름을 사용)
- 복장: 구체적인 아이템 이름 (`school uniform`, `business suit`, `lab coat`, `hoodie`, `maid outfit`)
- 필수 끝 태그: `full body, standing, white background, looking at viewer`

**네거티브 프롬프트 (권장):**

```
lowres, bad anatomy, bad hands, missing fingers, extra digits, fewer digits, worst quality, low quality, blurry, jpeg artifacts, cropped, multiple views, logo, too many watermarks
```

### 생성 절차

> **중요**: 생성 전에 반드시 `assets/prompt_realistic.txt` (리얼리스틱용) 또는 `assets/prompt.txt` (애니메용) 의 유무를 확인할 것. 캐시된 프롬프트가 존재하면 그것을 사용한다.

**스텝 1: 스타일 판정**

시스템의 이미지 스타일을 확인한다. `image_style` 은 보통 `realistic` (포토리얼리스틱) 또는 `anime` 중 하나.
프레임워크가 `generate_character_assets` 에 전달된 프롬프트의 스타일을 자동 감지·변환하지만, 처음부터 올바른 스타일의 프롬프트를 사용하는 것이 바람직하다.

- `assets/prompt_realistic.txt` 이 존재 → **그대로 사용** (최우선)
- `assets/prompt.txt` 이 존재 → 애니메 스타일이면 그대로 사용. 리얼리스틱이 필요하면 아래 규칙으로 변환
- 모두 존재하지 않음 → IDENTITY.md 의 외모 설정에서 새로 작성

**스텝 2: 프롬프트 작성**

**리얼리스틱 (사실적) 스타일의 경우:**

Fal.ai Flux Pro 로 포토리얼리스틱 이미지를 생성한다. 프롬프트는 Danbooru 태그가 아니라 자연 언어의 사진적 묘사를 사용한다.

```
professional photograph, studio lighting, high resolution, realistic, photorealistic, a young woman/man with {hair_description}, {eye_description}, {outfit_description}, full body, standing, plain white background, looking at viewer
```

변환 규칙 (애니메→리얼리스틱):

| 애니메 태그 | 리얼리스틱 묘사 |
|---|---|
| `masterpiece, best quality, ...` (품질 태그) | 삭제 (리얼리스틱 품질 태그로 치환) |
| `anime coloring, clean lineart, soft shading` | 삭제 |
| `1girl` | `a young woman` |
| `1boy` | `a young man` |
| `black hair, long hair, low ponytail` | `long black hair in a low ponytail` |
| `red eyes, narrow eyes` | `sharp red eyes` |

품질·스타일 태그 (앞에 부여): `professional photograph, studio lighting, high resolution, realistic, photorealistic`

**애니메 스타일의 경우:**

위 'NovelAI 프롬프트로의 변환' 섹션의 규칙으로 애니메 태그를 작성한다.

**스텝 3: 생성 실행**

시스템 프롬프트의 '외부 도구' 섹션에 기재된 **image_gen** (`generate_character_assets`) 의 사용 방법에 따라 호출한다.

인수:
- `prompt`: 스텝 2에서 작성한 프롬프트
- `negative_prompt`: 권장 네거티브 프롬프트
- `anima_dir`: 대상 Anima 의 디렉토리 (자신이면 자신의, 타인이면 타인의)
- `steps` 은 **지정하지 않는다** (기본값으로 전체 스텝이 실행된다)

**리얼리스틱 시 생성 결과:**
   - `avatar_fullbody_realistic.png` — 전신 사진 (Fal Flux Pro)
   - `avatar_bustup_realistic.png` — 버스트업 사진 (Flux Kontext)
   - 표정 변형: `avatar_bustup_{emotion}_realistic.png`
   - `icon_realistic.png` — 아이콘
   - 치비 캐릭터·3D 모델·리깅·애니메이션은 생성하지 않는다

**애니메 시 생성 결과:**
   - `avatar_fullbody.png` — 전신 서 있는 그림 (NovelAI V4.5)
   - `avatar_bustup.png` — 버스트업 (Flux Kontext)
   - `avatar_chibi.png` — 치비 캐릭터 (Flux Kontext)
   - `avatar_chibi.glb` — 3D 모델 (Meshy Image-to-3D)
   - `avatar_chibi_rigged.glb` — 리그 포함 3D 모델 (Meshy Rigging)
   - 애니메이션 (Meshy Animations)

생성에 실패한 스텝이 있으면 오류를 기록하고, 성공한 것만 사용한다.
