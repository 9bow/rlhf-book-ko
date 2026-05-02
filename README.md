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
│   ├── policy_gradients/   # PPO, REINFORCE, GRPO, RLOO
│   ├── reward_models/      # 선호도 RM, ORM, PRM 학습
│   ├── direct_alignment/   # DPO 및 변형
│   └── rejection_sampling/ # 최선-N 거부 샘플링
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

- 정책 경사 방법 (PPO, REINFORCE, GRPO, RLOO 등)
- 보상 모델 학습 (선호도 RM, ORM, PRM)
- 직접 정렬 방법 (DPO 및 변형)
- 거부 샘플링 (최선-N)

> - Policy gradient methods (PPO, REINFORCE, GRPO, RLOO, etc.)
> - Reward model training (preference RM, ORM, PRM)
> - Direct alignment methods (DPO and variants)
> - Rejection sampling (best-of-N)

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

- 코드: [MIT](LICENSE-CODE)
- 챕터: [CC-BY-NC-SA-4.0](LICENSE-CHAPTERS)

> - Code: [MIT](LICENSE-CODE)
> - Chapters: [CC-BY-NC-SA-4.0](LICENSE-CHAPTERS)

## 기여자

> ## Contributors

이 프로젝트의 유일한 '저자'이자 창작자로서 모든 공을 인정받고 있지만, 초기 독자들의 많은 기여 덕분에 행운이었습니다. 이들의 기여는 편집 과정을 크게 가속화했고, 책에 의미 있는 내용을 직접 추가해 주었습니다. 실질적인 기여자들에게는 책의 무료 사본을 기꺼이 보내드리며, 인터넷의 선한 의지가 예상치 못한 방식으로 보답해 줄 것을 기대합니다.

> Where I get the credit as the sole "author" and creator of this project, I've been super lucky to have many contributions from early readers. These have massively accelerated the editing progress and flat-out added meaningful content to the book. I'm happy to send substantive contributors free copies of the book and expect the internet goodwill to pay them back in unexpected ways.

모든 [기여자](https://github.com/natolambert/rlhf-book/graphs/contributors)를 확인하세요.

> See all [contributors](https://github.com/natolambert/rlhf-book/graphs/contributors).
