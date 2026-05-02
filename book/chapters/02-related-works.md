<!--
  Copyright (c) 2025-2026 Nathan Lambert.
  Licensed under CC BY-NC-SA 4.0:
  https://creativecommons.org/licenses/by-nc-sa/4.0/
  Full license: https://github.com/natolambert/rlhf-book/blob/main/LICENSE-CHAPTERS
-->
---
prev-chapter: "소개"
prev-url: "01-introduction"
page-title: 주요 관련 연구
search-title: "2장: 주요 관련 연구"
next-chapter: "학습 개요"
next-url: "03-training-overview"
lectures:
  - video: "https://www.youtube.com/watch?v=o6l6tJQgUg4&list=PLL1tdVxB1CpVpEtMHxwuR4uI4Lxjw00_y&index=2"
    label: "강의 1: 개요 (1–3장)"
---

# 주요 관련 연구

RLHF와 관련 방법들은 매우 새롭습니다.
우리는 절차가 얼마나 최근에 공식화되었는지, 그리고 이 문서화의 얼마나 많은 부분이 학술 문헌에 있는지를 보여주기 위해 역사를 강조합니다.
이를 통해, RLHF는 매우 빠르게 발전하고 있음을 강조하고자 합니다. 따라서 이 장은 특정 방법들에 대한 불확실성을 표현하고 몇 가지 핵심 관행 주변의 세부 사항이 변할 수 있다는 기대를 담고 있는 책의 무대를 설정합니다.
그 외에도, 여기에 나열된 논문과 방법들은 왜 RLHF 파이프라인의 많은 부분이 현재의 모습인지를 보여주며, 일부 기초 논문들은 현대 언어 모델과는 완전히 다른 응용 분야를 위한 것이었습니다.

이 장에서는 RLHF 분야를 오늘날의 위치로 이끈 핵심 논문과 프로젝트들을 자세히 설명합니다.
이것은 RLHF와 관련 분야에 대한 포괄적인 검토가 아니라, 오늘날에 이르게 된 과정의 시작점이자 재현입니다.
이것은 의도적으로 ChatGPT로 이어진 최근 연구에 집중합니다.
선호도로부터의 학습에 관한 RL 문헌에는 상당한 추가 연구가 있습니다 [@wirth2017survey].
더 철저한 목록을 위해서는 적절한 서베이 논문 [@kaufmann2023survey], [@casper2023open]을 참고하시기 바랍니다.

![이 장에서 다루는 RLHF의 주요 발전 타임라인으로, 선호도로부터의 RL 초기 연구부터 대규모 언어 모델에서의 RLHF 채택까지.](images/rlhf_timeline.png){#fig:rlhf_timeline}

## 기원에서 2018년까지: 선호도로부터의 RL

이 분야는 최근 심층 강화학습 (RL)의 성장으로 대중화되었으며 많은 대형 기술 기업들의 LLM 응용 연구로 더 넓게 발전했습니다.
그럼에도 불구하고, 오늘날 사용되는 많은 기법들은 선호도로부터의 RL에 관한 초기 문헌의 핵심 기법과 깊이 관련되어 있습니다.

현대 RLHF와 유사한 접근 방식을 가진 첫 번째 논문 중 하나는 *TAMER*였습니다.
*TAMER: Training an Agent Manually via Evaluative Reinforcement*는 인간이 에이전트의 행동을 반복적으로 평가하여 보상 모델을 학습시키고, 이를 행동 정책 (policy)을 학습하는 데 사용하는 접근법을 제안했습니다 [@knox2008tamer].
동시기 또는 그 직후의 다른 연구들은 이점 함수 (advantage function)를 조정하기 위해 인간 피드백(긍정 및 부정 모두)이 사용되는 행위자-비평자 (actor-critic) 알고리즘 COACH를 제안했습니다 [@macglashan2017interactive].

주요 참고문헌인 Christiano et al. 2017은 Atari 게임 내 에이전트 궤적 (trajectory) 간의 선호도에 적용된 RLHF의 응용입니다 [@christiano2017deep].
RLHF를 소개한 이 연구는 RL 에이전트가 처음부터 학습하여 인기 있는 비디오 게임을 해결할 수 있음을 보여준 DeepMind의 심층 Q-네트워크 (DQN) 기반 강화학습의 기초 연구 이후 곧 이어졌습니다.
이 연구는 궤적 사이에서 선택하는 인간이 일부 도메인에서 환경과 직접 상호작용하는 것보다 더 효과적일 수 있음을 보여줍니다. 이것은 몇 가지 영리한 조건을 사용하지만, 그럼에도 인상적입니다.

![Christiano et al. (2017)의 핵심 RLHF 루프: 보상 예측기는 궤적 세그먼트 비교로부터 비동기적으로 학습되며, 에이전트는 예측된 보상을 최대화합니다.](images/rlhf_schematic.png){#fig:rlhf_schematic width=66%}

이 방법은 더 직접적인 보상 모델링 [@ibarz2018reward]으로 확장되었으며, 초기 RLHF 연구 내에서 딥러닝의 채택은 1년 후 신경망 모델을 사용한 TAMER 확장으로 마무리되었습니다 [@warnell2018deep].

이 시대는 전환하기 시작했는데, 일반적인 개념으로서의 보상 모델이 단순히 RL 문제를 해결하기 위한 도구가 아니라 정렬을 연구하는 방법으로 제안되면서였습니다 [@leike2018scalable].

## 2019년에서 2022년까지: 언어 모델에서의 인간 선호도로부터의 RL

인간 피드백 기반 강화학습은, 초기에는 종종 인간 선호도로부터의 강화학습이라고도 불렸는데, 점점 더 대규모 언어 모델 확장에 집중하는 AI 랩들이 빠르게 채택했습니다.
이 연구의 상당 부분은 2019년의 GPT-2와 2020년의 GPT-3 사이에 시작되었습니다.
2019년의 가장 초기 연구인 *Fine-Tuning Language Models from Human Preferences*는 현대 RLHF 연구 및 이 책에서 다룰 내용과 놀라운 유사점이 많습니다 [@ziegler2019fine].
보상 모델 학습, KL 거리, 피드백 다이어그램 등과 같은 많은 표준 용어들이 이 논문에서 공식화되었으며 -- 최종 모델의 평가 태스크와 능력만 오늘날 사람들이 하는 것과 달랐습니다.
여기서부터 RLHF는 다양한 태스크에 적용되었습니다.
중요한 예시로는 일반 요약 [@stiennon2020learning], 책의 재귀적 요약 [@wu2021recursively], 지시 따르기 (InstructGPT) [@ouyang2022training], 브라우저 지원 질의응답 (WebGPT) [@nakano2021webgpt], 인용을 통한 답변 지원 (GopherCite) [@menick2022teaching], 그리고 일반 대화 (Sparrow) [@glaese2022improving]가 있습니다.

응용 외에도, 많은 기초 논문들이 RLHF의 미래를 위한 핵심 영역들을 정의했습니다:

1. 보상 모델 과최적화 [@gao2023scaling]: RL 최적화 도구가 선호도 데이터로 학습된 모델에 과적합할 수 있는 능력,
2. 정렬을 위한 일반적인 연구 영역으로서의 언어 모델 [@askell2021general], 그리고
3. 레드 팀 (red teaming) [@ganguli2022red] -- 언어 모델의 안전성을 평가하는 과정.

챗 모델에 RLHF를 적용하기 위한 개선 작업이 계속되었습니다.
Anthropic은 Claude의 초기 버전에 이를 광범위하게 사용했으며 [@bai2022training], 초기 RLHF 오픈 소스 도구들이 등장했습니다 [@ramamurthy2022reinforcement], [@havrilla-etal-2023-trlx], [@vonwerra2022trl].

## 2023년부터 현재까지: ChatGPT 시대

ChatGPT의 발표는 학습에서 RLHF의 역할에 대해 매우 명확했습니다 [@openai2022chatgpt]:

> 우리는 InstructGPT와 동일한 방법을 사용하여 인간 피드백 기반 강화학습 (RLHF)으로 이 모델을 학습시켰으며, 데이터 수집 설정에서 약간의 차이가 있습니다.

그 이후로, RLHF는 선도적인 언어 모델들에 광범위하게 사용되어 왔습니다.
Anthropic의 헌법적 AI (Constitutional AI)인 Claude [@bai2022constitutional], Meta의 Llama 2 [@touvron2023llama]와 Llama 3 [@dubey2024llama], Nvidia의 Nemotron [@adler2024nemotron], Ai2의 Tülu 3 [@lambert2024t], 그리고 더 많은 모델에서 사용된 것으로 잘 알려져 있습니다.

오늘날, RLHF는 더 넓은 선호도 미세조정 (PreFT) 분야로 성장하고 있으며, 중간 추론 단계를 위한 과정 보상 모델 (PRM) [@lightman2023let] (5장에서 다룸), 직접 선호도 최적화 (DPO)에서 영감을 받은 직접 정렬 알고리즘 [@rafailov2024direct] (8장에서 다룸), 코드 또는 수학의 실행 피드백으로부터의 학습 [@kumar2024training], [@singh2023beyond] 및 OpenAI의 o1 [@openai2024o1]에서 영감을 받은 다른 온라인 추론 방법들 (7장에서 다룸)과 같은 새로운 응용들을 포함합니다.
