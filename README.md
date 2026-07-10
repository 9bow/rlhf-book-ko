# RLHF Book

인간 피드백 기반 강화학습에 대한 종합 가이드 (그리고 언어 모델 사후 학습에 대한 폭넓은 입문서).

> A comprehensive guide to Reinforcement Learning from Human Feedback (and a broad introduction to post-training language models).

**[한국어 온라인 읽기](https://9bow.github.io/rlhf-book-ko/)** | **[PDF 다운로드](https://9bow.github.io/rlhf-book-ko/book.pdf)** | **[EPUB 다운로드](https://9bow.github.io/rlhf-book-ko/book.epub)**

**[원문 읽기](https://rlhfbook.com)** | **인쇄본 주문: [Manning](https://hubs.la/Q03Tc3cf0) 또는 [Amazon](https://amzn.to/4cwCDJQ)** | **[Discord 커뮤니티](https://discord.gg/yz5AwK4gBR) 참여**

> **[Read online](https://rlhfbook.com)** | **Order print on [Manning](https://hubs.la/Q03Tc3cf0) or [Amazon](https://amzn.to/4cwCDJQ)** | **Join [Discord Community](https://discord.gg/yz5AwK4gBR)**

이 책은 ChatGPT 이후 언어 모델의 급속한 발전 속에서 개방형 모델의 최전선에서 일하며 쌓은 모든 지식을 오픈소스로 공개하려는 시도입니다.
처음 시작했을 때, 거부 샘플링과 같이 이미 확립된 방법론들조차 표준적인 참고 자료가 없었습니다.
반면, 모델을 더 친근하게 만드는 산업 실무 — 통칭 '캐릭터 훈련' — 에는 공개 연구가 전무했습니다.
이를 문서화하고, 기본기를 다지고, 참고 자료를 꼼꼼히 정리하는 것이 (AI 슬롭의 시대에) 사람들에게 훌륭한 출발점이 될 것임이 분명했습니다.

> This book is my attempt to open-source all the knowledge I've gained working at the frontier of open models in the post-ChatGPT take off of language models.
> When I started, many established methods like rejection sampling had no canonical reference.
> On the other side, industry practices to make the models more personable -- colloquially called Character Training -- had no open research.
> It was obvious to me that there would be payoff to documenting, learning the fundamentals, carefully curating the references (in an era of AI slop), and everything in between would be a wonderful starting point for people.

지금은 코드를 추가하며 배우고 싶은 사람들의 거점으로 삼고 있습니다.
코딩 어시스턴트를 활용해 질문하세요.
현실 세계가 중요하니 실물 책도 구매하세요.
당신에게 맞춤화된 AI 출력물을 읽으세요.

> Today, I'm adding code and seeing this as a home base for people who want to learn.
> You should use coding assistants to ask questions.
> You should buy the physical book because the real world matters.
> You should read the specific AI outputs tailored to you.

앞으로 오픈소스 슬라이드 덱 등 더 많은 교육 자료를 추가할 계획입니다.
결국, 인간의 선호도를 측정하는 것이 얼마나 어려운지를 생각하면 RLHF는 결코 완전히 해결된 문제가 되지 않을 것입니다.

> In the future I want to build more education resources to this, such as open source slide decks and more ways to learn.
> In the end, with how impossible it is to measure human preferences, RLHF will never be a solved problem.

읽어주셔서 감사합니다.
피드백을 남기거나 커뮤니티에 참여해 주셔서 감사합니다.

> Thank you for reading.
> Thank you for contributing any feedback or engaging with the community.

-- Nathan Lambert, @natolambert

## 한국어 번역 (Korean Translation)

이 저장소는 [natolambert/rlhf-book](https://github.com/natolambert/rlhf-book)의 한국어 번역본입니다.

- **번역자**: [9bow](https://github.com/9bow)
- **번역 도구**: Claude Sonnet 4.6 (Anthropic)
- **원본**: [natolambert/rlhf-book](https://github.com/natolambert/rlhf-book) by Nathan Lambert
- **라이선스**: 원본과 동일하게 [CC-BY-NC-SA-4.0](LICENSE-CHAPTERS) 적용

번역 오류나 개선 사항은 [이슈](https://github.com/9bow/rlhf-book-ko/issues)로 제보해 주세요.

---

## 저장소 구조

> ## Repository Structure

```
rlhf-book/
├── book/                   # 책 소스 및 빌드 파일
│   ├── chapters/           # 마크다운 소스 (01-introduction.md 등)
│   ├── images/             # 챕터에서 참조하는 이미지
│   ├── assets/             # 브랜드 에셋 (표지, 로고)
│   ├── templates/          # Pandoc 템플릿 (HTML, PDF, EPUB)
│   ├── scripts/            # 빌드 유틸리티
│   └── data/               # 라이브러리 데이터
├── code/                   # 참조 구현
│   ├── instruction_tuning/  # 지시 미세조정(SFT)
│   ├── policy_gradients/   # PPO, REINFORCE, GRPO, RLOO
│   ├── reward_models/      # 선호도 RM, ORM, PRM 학습
│   ├── direct_alignment/   # DPO 및 변형
│   ├── rejection_sampling/ # 최선-N 거부 샘플링
│   └── distillation/       # 온-정책 증류(SDPO)
├── diagrams/               # 다이어그램 소스 파일
│   ├── scripts/            # Python 생성 스크립트
│   ├── tikz/               # LaTeX/TikZ 소스
│   └── specs/              # YAML 명세
├── teach/                  # 교육 자료 (강좌, 슬라이드)
├── build/                  # 생성된 출력물 (git 무시)
└── Makefile                # 빌드 시스템
```

## 코드 라이브러리

> ## Code Library

`code/`에 있는 RLHF 알고리즘 참조 구현:

> Reference implementations for RLHF algorithms in `code/`:

- 정책 그래디언트 방법 (PPO, REINFORCE, GRPO, RLOO 등)
- 지시 미세조정 (SFT)
- 보상 모델 학습 (선호도 RM, ORM, PRM)
- 직접 정렬 방법 (DPO 및 변형)
- 거부 샘플링 (최선-N)
- 온-정책 증류 (SDPO)

> - Policy gradient methods (PPO, REINFORCE, GRPO, RLOO, etc.)
> - Instruction tuning (SFT)
> - Reward model training (preference RM, ORM, PRM)
> - Direct alignment methods (DPO and variants)
> - Rejection sampling (best-of-N)
> - On-policy distillation (SDPO)

설정 및 사용법은 [code/README.md](code/README.md)를 참고하세요.

> See [code/README.md](code/README.md) for setup and usage.

## 책 소스

> ## Book Source

책 소스 파일은 `book/`에 있습니다. 로컬에서 빌드하려면:

> Book source files are in `book/`. Build locally:

```bash
make html   # HTML 사이트 빌드
make pdf    # PDF 빌드 (LaTeX 필요)
make epub   # EPUB 빌드
```

> ```bash
> make html   # Build HTML site
> make pdf    # Build PDF (requires LaTeX)
> ```

자세한 빌드 방법은 [book/README.md](book/README.md)를 참고하세요.

> See [book/README.md](book/README.md) for detailed build instructions.

## 다이어그램

> ## Diagrams

`diagrams/` 디렉토리에는 책에 사용된 도형의 소스 파일이 있습니다. 프레젠테이션, 블로그 포스트, 또는 개인 학습 자료로 재사용할 수 있도록 설계되었습니다. 다음 명령으로 생성하세요:

> The `diagrams/` directory contains source files for figures used in the book. These are designed to be reusable for presentations, blog posts, or your own learning materials. Generate them with:

```bash
cd diagrams && make all
```

## 인용

> ## Citation

이 책을 인용할 때는 다음 형식을 사용하세요:

> To cite this book, please use the following format:

```bibtex
@book{rlhf2026lambert,
  author       = {Nathan Lambert},
  title        = {Reinforcement Learning from Human Feedback},
  year         = {2026},
  publisher    = {Online},
  url          = {https://rlhfbook.com},
}
```

## 라이선스

> ## License

- `book/chapters`: [CC-BY-NC-SA-4.0](LICENSE-CHAPTERS)
- 그 외 전체(`code/`, `diagrams/`, `scripts/` 등): [MIT](LICENSE-CODE)
- `book/images/`의 일부 이미지는 라이선스가 명시되지 않은 사진 또는 스크린샷일 수 있습니다.

> - `book/chapters`: [CC-BY-NC-SA-4.0](LICENSE-CHAPTERS)
> - everything else (`code/`, `diagrams/`, `scripts/`, etc.): [MIT](LICENSE-CODE)
> - note that some images in `book/images/` are unlicensed photos or screenshots.

## 기여자

> ## Contributors

이 프로젝트의 유일한 '저자'이자 창작자로서 모든 공을 인정받고 있지만, 초기 독자들의 많은 기여 덕분에 행운이었습니다. 이들의 기여는 편집 과정을 크게 가속화했고, 책에 의미 있는 내용을 직접 추가해 주었습니다. 실질적인 기여자들에게는 책의 무료 사본을 기꺼이 보내드리며, 인터넷의 선한 의지가 예상치 못한 방식으로 보답해 줄 것을 기대합니다.

> Where I get the credit as the sole "author" and creator of this project, I've been super lucky to have many contributions from early readers. These have massively accelerated the editing progress and flat-out added meaningful content to the book. I'm happy to send substantive contributors free copies of the book and expect the internet goodwill to pay them back in unexpected ways.

모든 [기여자](https://github.com/natolambert/rlhf-book/graphs/contributors)를 확인하세요.

> See all [contributors](https://github.com/natolambert/rlhf-book/graphs/contributors).

참고: 저자가 더 이상 접근할 수 없는 Ai2 이메일에 커밋을 연결해 둔 실수 때문에, 커밋 히스토리에서 저자 기여 추적 대부분이 사라졌다고 합니다.

> Note: *because I made the mistake of associating my commits with my Ai2 email, which I no longer have access to, the commit history lost most of my tracking, RIP!*

### 번역

> ### Translations

독자들은 각자의 저장소에서 책의 비공식 번역을 유지하고 있습니다.
이 번역들은 커뮤니티 프로젝트입니다. 공식 인쇄판 및 전문 번역과는 독립적이며, 이 책에 대한 출처 표시와 함께 동일한 CC-BY-NC-SA 라이선스로 공개됩니다.

> Readers maintain unofficial translations of the book in their own repositories.
> These are community projects — independent of the official print editions and their professional translations — released under the same CC-BY-NC-SA license with attribution back to this book:

- 简体中文 (Simplified Chinese): [jweihe/RLHF-book-Chinese](https://github.com/jweihe/RLHF-book-Chinese)

번역을 추가하려면 별도 저장소에 유지하고(번역본은 이 저장소에 병합하지 않습니다), 위 라이선스 조건을 따르며, 커뮤니티 번역임을 명확히 표시한 뒤 이 목록과 홈페이지 Ecosystem 섹션(`book/templates/html.html`)에 추가하는 PR을 열어 주세요.

> To add yours: keep it in your own repo (translations are not merged here), follow the license terms above, label it clearly as a community translation, then open a PR adding it to this list and to the homepage Ecosystem section (`book/templates/html.html`).

### AI 사용 정책

> ### AI Use Policy

저자는 이 책의 편집과 제작을 돕는 데 AI를 어떻게 사용했는지, 그리고 기여자에게 무엇을 기대하는지 명확히 문서화하고자 했습니다.
이 책은 AI 모델이 지식 노동의 유용한 도구에서 필수 도구로 전환되던 흥미로운 시기에 작성되었습니다.

> I wanted to clearly document how I used AI to aid in the editing and creation of this book (and my expectations for contributors).
> This book was written at an interesting time, when AI models transitioned from useful to essential as tools for knowledge work.

책의 핵심부는 언어 모델이 논픽션 글쓰기에 거의 쓸모없게 느껴지던 시기에 작성되었습니다. 대략 처음 10개 장이 여기에 해당하며, 저자가 사후 학습을 배워가며 남긴 개인 노트에서 출발했습니다.
초안은 거의 전적으로 수작업이었고(오탈자까지 git 히스토리에 남아 있습니다), 다른 장과 부록의 상당 부분은 저자의 개인 뉴스레터인 [interconnects.ai](https://interconnects.ai/)의 콘텐츠를 직접 각색했습니다.
이 글은 저자의 목소리가 강하게 남아 있으며, 직관 전달을 유지하기 위해 아주 가벼운 AI 편집만 사용했습니다.
수학과 코드가 적은 장일수록 AI 사용도 적었습니다.

> The core of this book was written when language models felt borderline useless for non-fiction writing; this is roughly the first 10 chapters of the book -- it was my personal notes as I learned post-training.
> The first draft was almost entirely manual (typos and all are in the git history).
> Much of the other chapters and the appendices were adapted directly from content on [interconnects.ai](https://interconnects.ai/), my personal newsletter.
> This writing is very high-voice and uses the lightest of AI editing to maintain the communication of intuitions.
> The less math and code in a chapter, the less I used AI.

편집 과정의 기본 워크플로우는 인간 편집자가 제안한 수정 목록과 문맥을 Claude Code에 전달하고, 각 항목을 하나씩 적용하도록 요청하는 방식이었습니다.
이 형식에서는 저자가 문맥을 읽고 수정 방향을 작성했습니다.
단순한 오탈자, 명백한 오류, 또는 아주 짧은 문구 수정은 Claude가 직접 반영할 수 있었습니다.
더 복잡한 문장 수정은 보통 저자가 직접 여러 문장과 추가 문구를 다시 작성해 만들었습니다.
또한 저자는 Cursor에서 글을 쓴 뒤 GitHub 작업을 Claude에게 맡기는 경우도 많았습니다.

> Through the editing, the default workflow I used was passing a list of suggested edits from a human editor to Claude Code with the context, and asking it to go one-by-one to apply various edits.
> In this format, I'd read the context and write a fix.
> In a case where the edit was a simple typo or blatant error or just a low number of words, Claude could directly make this edit.
> More complex language edits were crafted by me, normally with me re-writing various sentences and additions.
> I also often just write in Cursor and then ask Claude to handle GitHub for me.

저자의 글을 꾸준히 읽어온 독자라면, 이 책과 Interconnects의 차이는 이 저장소에서는 저자가 제안한 편집을 AI 에이전트가 적용하도록 했다는 점이고, 블로그에서는 그런 작업을 모두 수작업으로 한다는 점입니다.
이는 주로 규모와 복잡도 때문입니다.

> If you follow my writing closely, the difference between this book and Interconnects is that I let AI agents apply edits I suggested for me in this repo, whereas on my blog I make a point of doing all of that work manually.
> This is largely a function of scale and complexity.

수학이 많은 장에서는 모델이 LaTeX 방정식과 기본 코드 조각을 조작하는 데 매우 유용했습니다.
이 부분들은 AI 모델의 출력이 더 직접적으로 반영된 편입니다. 저자가 LaTeX를 모두 수작업으로 작성하면 상당한 시간이 걸렸기 때문입니다.
그 뒤 수학과 코드는 저자가 직접, 그리고 GPT-Pro 모델을 통한 검토로 한 번 더 확인했습니다.

> For the more math-heavy chapters, the models are unbelievably useful at manipulating LaTeX equations and basic code snippets.
> These sections are more direct outputs from the AI models, as me writing the LaTeX manually would take substantial time.
> Then, I would review the math and code an additional time manually and with the check of GPT-Pro models.

AI 모델은 `diagrams/`와 `code/`에서 훨씬 더 광범위하게 사용되었습니다. 저자는 이를 책의 핵심 내용 주변에서 최신 모델을 가지고 실험해 보는 일종의 놀이로 보았습니다.

> AI models were used much more extensively in `diagrams/` and `code/`, where I viewed these as a form of play with the latest models, around the substantial content of the book.

실물 책은 추가적이고 상당한 수준의 카피 편집을 거쳐 기술 교과서에 더 가까운 표준 문체로 다듬어졌습니다.

> The physical edition of the book went through additional, substantial copy editing that transitioned the voice to be more of a standard style for a technical textbook.

책에 새로 추가되는 내용은 당연히 사람이 먼저 작성해야 하며, 이후 AI로 편집할 수 있습니다. 이는 위에서 설명한 저자의 워크플로우를 반영합니다.
GitHub PR에 명백히 AI가 작성한 듯한 내용이 포함되어 있다면 거의 확실히 포함되지 않을 것입니다.

> All new additions to the book should obviously be written by humans first, and then can be edited with AI, as this reflects my workflow above.
> The presence of obvious AI-written content in a PR to GitHub will almost certainly result in it not being included.
