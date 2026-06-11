# 교육용 슬라이드

RLHF Book 강의 슬라이드이며 [colloquium](https://github.com/natolambert/colloquium)으로 빌드합니다.

Colloquium은 활발히 개발 중이므로, 콘텐츠와 함께 슬라이드 도구도 계속 개선될 수 있습니다.

## 강의 묶음

- **`course/`** — 책 전체를 다루는 입문 강좌(5~10개 강의, 작성 중)
- **`SALA-2026/`** — 예정된 발표

## 설정

저장소 루트에서 설치합니다(`teach` extra를 통해 GitHub의 `colloquium`을 가져옵니다):

```bash
uv sync --extra teach
```

`colloquium`이 빠르게 바뀌고 있으므로 `teach` extra는 의도적으로 GitHub 소스를 추적합니다.
슬라이드 명령을 직접 실행할 때는 새 환경에서도 `colloquium`을 사용할 수 있도록 `uv run --extra teach ...`를 사용하세요.

개발을 위해 로컬 클론에서 설치할 수도 있습니다:

```bash
uv pip install -e /path/to/colloquium
```

## 빌드

저장소 루트에서:

```bash
make teach
```

단일 강의만 빌드할 수도 있습니다:

```bash
uv run --extra teach colloquium build teach/course/lec1-chap1-3.md -o build/html/teach/course/lec1-chap1-3/
```

출력은 `build/html/teach/`에 생성됩니다.

## 라이브 미리보기

강좌 슬라이드는 `assets/...` 이미지 링크가 올바르게 해석되도록 생성된 HTML을 `teach/course/`에 둡니다:

```bash
uv run --extra teach python -c "from colloquium.serve import serve; serve('teach/course/lec5-chap7.md', port=8081, output_dir='teach/course')"
```

`http://localhost:8081/lec5-chap7.html`을 엽니다.

로컬 에셋을 가진 독립 발표는 발표 디렉터리에서 실행합니다:

```bash
cd teach/SALA-2026
uv run --extra teach colloquium serve talk.md
```

미리보기 URL을 공유하기 전에 예상 이미지 URL 하나 이상이 `200 image/...`를 반환하는지 확인하세요.

## 발표 에셋

각 발표의 이미지는 로컬 `assets/` 디렉터리(예: `teach/SALA-2026/assets/`)에 둡니다. 슬라이드에서는 상대 경로로 참조합니다:

```markdown
![description](assets/image.png)
```
