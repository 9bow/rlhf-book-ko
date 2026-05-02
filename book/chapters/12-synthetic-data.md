<!--
  Copyright (c) 2025-2026 Nathan Lambert.
  Licensed under CC BY-NC-SA 4.0:
  https://creativecommons.org/licenses/by-nc-sa/4.0/
  Full license: https://github.com/natolambert/rlhf-book/blob/main/LICENSE-CHAPTERS
-->
---
prev-chapter: "선호도 데이터"
prev-url: "11-preference-data"
page-title: 합성 데이터 및 CAI
search-title: "Chapter 12: 합성 데이터 및 CAI"
next-chapter: "도구 사용"
next-url: "13-tools"
---

# 합성 데이터 및 지식 증류

*인간 피드백* 기반 강화학습은 우리가 구축하는 모델에 인간의 영향력을 유지한다는 아이디어에 깊이 뿌리를 두고 있다.
첫 번째 모델이 RLHF로 성공적으로 학습되었을 때, 인간 데이터는 이런 방식으로 모델을 개선할 수 있는 *유일한* 실행 가능한 방법이었다.

인간만이 질문에 대한 학습을 위한 충분히 높은 품질의 응답을 만들 수 있었다.
인간만이 보상 모델을 학습시키기 위한 신뢰할 수 있고 구체적인 피드백 데이터를 수집할 수 있었다.

AI 모델이 더 좋아지면서, 이 가정은 빠르게 무너졌다.
훨씬 더 저렴하고 반복하기 쉬운 합성 데이터 (synthetic data)의 가능성은, RLHF가 주목의 중심이었던 것에서 더 넓은 "후처리 학습 (post-training)"이 모델을 형성한다는 개념으로의 확산을 가능하게 했다.
이 장은 합성 데이터가 RLHF 파이프라인의 여러 부분을 어떻게 그리고 왜 대체하거나 확장하고 있는지에 대한 간략한 개요를 제공한다.

합성 데이터에 대한 일반적인 비판 중 하나는 **모델 붕괴(model collapse)** -- 모델 자체의 생성물로 반복적으로 학습하면 효과적인 학습 분포를 점진적으로 좁힐 수 있다는 아이디어 [@shumailov2024ai]다.
다양성이 감소하면 희귀한 사실과 스타일이 과소 표현되고, 작은 실수가 반복을 거쳐 증폭되어 더 나쁜 일반화로 이어질 수 있다.
실제로, 이러한 실패는 주로 필터링되지 않은 반복적인 단일 모델 출력에 대한 자기 학습과 관련이 있다; 실제/인간 데이터를 혼합하거나, 다양한 교사를 사용하거나, 중복 제거 및 강력한 품질 필터를 적용하면 붕괴 체제를 대부분 피할 수 있다.
오늘날의 프론티어 학습 파이프라인에서, 증거는 합성 데이터가 붕괴 이야기의 가장 강력한 버전이 암시하는 재앙적 회귀 없이 규모에서 사용될 수 있고, 사용되어야 한다는 것을 시사한다 [@gerstgrasser2024model] [@feng2024beyond].

최고의 모델들은 최상의 성능에 도달하기 위해 **합성 데이터가 필요하다**.
현대 후처리 학습에서 합성 데이터는 학습의 많은 부분을 포함한다 -- 언어 모델은 시드 예시로부터 새로운 학습 프롬프트를 생성하고 [@wang2022self], 기존 프롬프트를 수정하고, 프롬프트에 대한 완성물을 생성하고 [@numina_math_7b], 선호도 데이터를 만들기 위한 AI 피드백을 제공하고 [@cui2023ultrafeedback], 완성물을 필터링하고 [@li2024superfiltering], 그 이상을 하기 위해 사용된다.
합성 데이터는 후처리 학습의 핵심이다.

합성 데이터가 이 정도 규모로 영향을 미칠 수 있게 된 것은 GPT-4 클래스 모델들과 함께 등장했다.
Llama 2와 GPT-3.5-Turbo와 같은 초기 언어 모델들에서는, 모델들이 데이터 파이프라인을 생성하거나 감독하는 데 충분히 신뢰할 수 없었다.
1-2년 내에, 언어 모델은 답변 생성에서 인간보다 훨씬 우수해졌다.
GPT-3.5에서 GPT-4 클래스 모델로의 전환에서, 모델이 LLM-as-a-judge 작업을 수행하는 능력도 등장했다.
GPT-4 또는 더 나은 모델들은 특정 콘텐츠에 대한 피드백이나 점수를 생성하는 데 훨씬 더 강건하고 일관적이다.

2022년 말 ChatGPT 출시 이후 몇 년에 걸쳐, 우리는 수많은 영향력 있는 합성 데이터셋을 보았다 -- 일부 예시: DPO 혁명에 불을 지핀 최초의 주목할만한 합성 선호도 데이터셋인 UltraFeedback [@cui2023ultrafeedback], 또는 2023년에 첫 번째 채팅 스타일 미세조정 데이터셋 중 하나인 Stanford Alpaca, Tülu 3 [@lambert2024t]의 기술 집중형(예: 수학, 코드, 지시 따르기) 합성 데이터셋, 또는 사고 모델을 학습시키기 위한 2025년의 OpenThoughts 3과 많은 다른 합성 추론 데이터셋 [@guha2025openthoughts].
오늘날 산업급 후처리 학습을 시작하기 위한 표준 참고 자료의 대부분은 위의 Tülu 3 또는 OpenThoughts 3과 같은 데이터셋을 포함하며, 훨씬 빠른 학습으로 인해 빠른 시작 가이드는 종종 Alpaca와 같은 더 작고 간단한 데이터셋으로 시작한다.

큰 변화는 또한 데이터셋 크기와 관련이 있으며, 미세조정 데이터셋은 프롬프트 수가 증가했고, Alpaca는 52K, OpenThoughts와 Tülu 3는 100만+ 샘플이며, 응답의 길이도 증가했다.
더 긴 응답과 더 많은 프롬프트는 Alpaca 데이터셋이 약 1,000만 학습 토큰 규모인 반면, Tülu는 약 5억에서 50배 더 크고, OpenThoughts 3은 약 100억 토큰 규모로 더 크다.

이러한 전환 전반에 걸쳐, 합성 데이터는 파이프라인 전체에서 인간 데이터를 균일하게 대체하지는 않았다.
**지시 데이터 (SFT)** 의 경우, 합성 생성이 크게 승리했다 -- 더 강한 모델로부터의 지식 증류 (knowledge distillation)는 이제 대부분의 인간 작가들이 규모에서 제공할 수 있는 것보다 더 높은 품질의 완성물을 생성한다 (가장 어려운 프론티어 추론 문제들에서 일부 예외 있음).
**RLHF에서의 선호도 데이터** 의 경우, 그림은 더 복잡하다: 학술 연구는 합성 선호도 데이터가 비슷하게 수행된다는 것을 보여주지만, 프론티어 연구소들은 여전히 인간 선호도 데이터를 경쟁적 해자로 여긴다.
**평가** 의 경우, 분열은 다른 양상을 취한다: LLM-as-a-judge는 비용 효율적으로 모델 출력의 *채점*을 확장하지만, 기본 벤치마크와 ground-truth 레이블은 여전히 인간이 만들어야 한다.
패턴은 합성 데이터가 모델이 인간의 신뢰성을 초과하는 곳에서 지배하는 반면, 인간은 역량 프론티어, ground truth 설정, 그리고 학습 안내에서 여전히 필수적이라는 것이다.

지식 증류라는 용어는 언어 모델에서 합성 데이터의 역할에 관한 논의에서 가장 강력한 형태였다.
지식 증류라는 용어는 딥러닝 문헌의 교사-학생 지식 증류 [@hinton2015distilling]의 기술적 정의에서 비롯되었다.

![전통적인 지식 증류 (knowledge distillation)는 KL 발산 손실을 사용하여 더 작은 학생 모델이 더 큰 교사 모델의 소프트 확률 분포를 맞추도록 학습시킨다. 두 모델이 동시에 동일한 입력을 처리하며, 온도 스케일링($\tau > 1$)은 분포를 부드럽게 하여 클래스 관계에 대한 더 많은 정보를 드러낸다.](images/knowledge_distillation_tikz.png){#fig:knowledge-distillation}

지식 증류는 구어적으로 더 강한 모델의 출력을 사용해 더 작은 모델을 학습시키는 것을 의미한다.

![LLM 후처리 학습에서의 합성 데이터 생성: 프롬프트가 강한 모델을 통해 완성물을 생성하도록 전달되며, 이것이 쌍을 이루어 학습 데이터셋을 만든다. 이 데이터셋은 표준 지도 학습을 통해 더 작은 모델을 미세조정하는 데 사용된다. 더 복잡한 파이프라인은 완성물을 편집하거나, 선호도 쌍을 생성하거나, 품질을 필터링하는 여러 모델을 포함할 수 있다.](images/synthetic_data_distillation_tikz.png){#fig:synthetic-data-generation}
후처리 학습에서, 이 일반적인 지식 증류 개념은 두 가지 일반적인 형태를 취한다:

1. 후처리 학습 과정의 광범위한 부분에 걸쳐 사용할 데이터 엔진으로: 지시에 대한 완성물, 선호도 데이터(또는 헌법적 AI), 또는 RL을 위한 검증.
2. 더 강한 모델에서 더 약한 모델로 특정 기술을 전달하기 위해, 이는 종종 수학적 추론이나 코딩과 같은 특정 기술에 대해 수행된다.

첫 번째 전략은 언어 모델이 다양한 작업에 대한 답변 작성에서 인간보다 더 신뢰할 수 있게 됨에 따라 인기를 얻었다.
GPT-4 클래스 모델은 수학과 코드와 같은 복잡한 작업에 대해 더 강한 모델의 지식 증류를 사용하는 범위를 확장했다 (위에서 언급한 것처럼).
여기서, 지식 증류는 연구소가 공개적으로 출시되지 않고 더 강한 모델을 만들기 위해 내부적으로만 사용되는 Claude Opus나 Gemini Ultra와 같은 대형 내부 모델을 학습시키는 모델 스위트를 갖는 것을 동기화한다.
오픈 모델에서, 일반적인 관행은 폐쇄 API 모델로부터 학습 데이터를 더 작고 공개적으로 사용 가능한 가중치로 증류하는 것이다 [@tunstall2023zephyr].
이를 위해, 교사 모델에서 고품질 프롬프트를 큐레이션하고 응답을 필터링하는 것이 성능을 극대화하는 데 결정적이다.

더 작은 언어 모델로 특정 기술을 전달하는 것은 동일한 지식 증류 원칙을 사용한다 -- 학습을 위해 가능한 최상의 데이터를 얻는 것이다.
여기서, 많은 논문들이 더 강한 모델에서 제한된 데이터셋을 사용해 정렬 [@zhou2023lima], 수학적 추론 [@shridhar2023distilling] [@hsieh2023distilling], 그리고 테스트 타임 스케일링 [@muennighoff2025s1]을 개선하는 것을 연구했다.

## 헌법적 AI와 AI 피드백

RLHF의 폭발적 성장 이후 곧, AI 피드백 기반 강화학습 (RLAIF)이 AI가 파이프라인의 인간 데이터 부분을 근사화하고 실험이나 진전을 가속화할 수 있는 대안적 접근법으로 등장했다.
AI 피드백은 일반적으로 특정 입력의 품질을 설명하는 데이터를 보강하거나 생성하기 위해 AI를 사용하는 더 큰 기법들의 집합이며 (다른 학습 접근법이나 평가에 사용될 수 있는), 이는 쌍별 선호도로 시작되었다 [@lee2023rlaif] [@sharma2024critical] [@castricato2024suppressing].
인간 피드백을 완전히 대체하거나 보강하기 위해 RLAIF를 사용하는 많은 동기들이 있다.
RLHF 과정 내에서, AI 피드백은 선호도 데이터 수집과 관련 보상 모델 학습 단계에서의 역할로 가장 잘 알려져 있다 (헌법적 AI는 특정 유형의 구현이다).
이 장에서는 일반적인 AI 피드백과 RLHF 학습 파이프라인에서 이를 사용하는 이 특정 방법에 초점을 맞추며, 이 책 후반에서 합성 데이터를 이해하거나 사용하는 더 많은 방법을 다룬다.

AI 피드백이 성숙해지면서, 그 응용은 단순히 인간 선호도 레이블을 대체하는 것을 넘어 확장되었다.
더 저렴한 선호도 데이터 수집을 가능하게 한 동일한 LLM-as-a-judge 인프라는 확장 가능한 평가도 가능하게 했으며 (16장 참조), 더 최근에는 검증 가능한 답변이 없는 영역으로 RL 학습을 확장하는 루브릭 기반 보상도 가능하게 했다 -- 이 장 후반에서 탐구하는 프론티어.

## AI와 인간 피드백 데이터의 균형

AI 모델은 특정 양의 피드백을 생성하는 데 인간보다 훨씬 저렴하며, 이 글을 쓰는 시점에서 단일 인간 선호도 데이터 비용이 약 $1 이상(또는 프롬프트당 $10 이상)인 반면, GPT-4o와 같은 프론티어 AI 모델을 사용하는 AI 피드백은 $0.01 미만이다.
이를 넘어, 인간 노동의 비용은 대략 일정하게 유지되는 반면, 이러한 작업에서 선도 모델의 성능은 계속 향상되면서 성능 대비 가격은 감소하고 있다.
이 비용 차이는 RLHF 방법론 실험 시장을 이전에는 가격 때문에 배제되었던 전체 인구층에게 개방한다.

가격 외에도, AI 피드백은 인간 피드백과 다른 *트레이드오프*를 성능에 도입하며, 이는 더 넓은 문헌에서 여전히 조사되고 있다.
AI 피드백은 우리가 학습하는 언어 모델의 평가에서 훨씬 더 두드러지며, 낮은 가격으로 인해 인간 데이터의 비용(또는 시간 지연)이 비실용적인 다양한 대규모 작업에 걸쳐 사용될 수 있다.
이러한 주제들은 모두 깊이 얽혀 있다 -- AI 피드백 데이터는 평가를 위해서도 인간 데이터를 완전히 대체하지 않을 것이며, 평가를 위한 AI 피드백의 양은 학습보다 훨씬 더 많을 것이다 왜냐하면 훨씬 더 많은 사람들이 모델을 학습시키는 것보다 평가하기 때문이다.

AI 피드백 데이터가 인간 데이터를 능가하는 정확한 영역과 응용 -- 즉 채팅, 안전, 추론, 수학 등 -- 은 완전히 확립되지 않았다.
RLAIF에서의 일부 초기 연구는 AI 피드백이 인간 데이터를 완전히 대체할 수 있다고 주장하며, 효과적인 대체제로 홍보한다 [@lee2023rlaif] 그리고 특히 채팅 작업에서만 평가될 때 [@cui2023ultrafeedback] [@yuan2025selfrewardinglanguagemodels].
ChatGPT 이후 RLHF를 연구하는 초기 문헌은 다양한 영역에서 도움이 되는 어시스턴트로 행동하는 모델의 "정렬"에 초점을 맞춘 좁은 평가 스위트를 가지고 있었다 (17장에서 더 논의됨).
후기 연구는 더 미묘한 그림을 제시하며, 일부 추론 작업을 포함하는 더 넓은 평가 세트에서의 최적 균형점이 정확하게 레이블링하기 어려운 도전적인 데이터 포인트들을 인간에게 라우팅하면서 대부분의 데이터는 AI 피드백으로 보내는 것을 포함한다 [@miranda2024hybrid] [@xu2025rlthf].
RLHF의 더 넓은 영역에서 인간과 AI 피드백 데이터 사이의 균형에 관한 집중적인 연구는 없지만, RLHF가 일반적으로 이 광범위한 평가 스위트를 개선할 수 있음을 보여주는 많은 기술 보고서가 있으며, DPO를 사용하는 것들, 예를 들어 Ai2의 Tülu 3 [@lambert2024t] & Olmo 3 [@teamolmo2025olmo3], 또는 HuggingFace의 SmolLM 3 [@bakouch2025smollm3], 그리고 온라인 RLHF 파이프라인을 사용하는 것들, 예를 들어 Scale AI의 인간 선호도 데이터와 LLM 기반 피드백 혼합을 사용하는 Nvidia의 연구 (helpsteer 연구 라인을 통해 [@wang2024helpsteer] [@wang2024helpsteer2] [@wang2024helpsteer2p] [@wang2025helpsteer3]): Nemotron Nano 3 [@nvidia2025nemotron3nano], Nemotron-Cascade [@wang2025nemotron], 또는 Llama-Nemotron 추론 모델 [@bercovich2025llamanemotron].

전반적으로, AI 피드백 및 관련 방법이 분야에 명백히 매우 유용하지만, 인간 데이터가 이러한 저렴한 대안들로 완전히 대체되지 않은 것은 분명하다.
많은 가설이 존재하지만, 인간 데이터가 실제 제품 환경에서 모델의 더 세밀한 제어를 허용하는지 또는 캐릭터 학습과 같은 더 새로운 학습 방법들에 대해 (17장에서 다루는, 모델의 성격을 정밀하게 제어할 수 있는 신흥 기법들) 연구되지 않았다.
시작하는 사람들에게, AI 피드백은 첫 번째 시도가 되어야 하지만, 더 큰 운영으로 확장하는 파이프라인의 경우 인간 피드백을 포함하도록의 결국적인 전환이 likely하다.

RLAIF라는 용어는 Anthropic의 연구 *헌법적 AI: AI 피드백으로부터의 무해성* [@bai2022constitutional]에서 도입되었으며, 이는 논문 제목의 두 방법(헌법적 AI와 AI 피드백) 사이의 관계에 대해 AI 커뮤니티에서 초기 혼란을 야기했다.
헌법적 AI (CAI) 논문의 출시와 RLAIF의 공식화 이후, RLAIF는 후처리 학습 및 RLHF 문헌 내의 기본 방법이 되었다 -- 쉽게 열거할 수 있는 것보다 훨씬 많은 예시들이 있다.
관계는 CAI가 더 넓은 RLAIF 분야를 시작한 예시로 이해되어야 한다.

인간 데이터와 AI 피드백 데이터의 차이에 대한 경험 법칙은 다음과 같다:

1. 인간 데이터는 고노이즈(high-noise), 저편향(low-bias)이다. 이는 데이터의 수집과 필터링이 더 어려울 수 있지만, 잘 다뤄지면 매우 신뢰할 수 있는 신호를 제공할 것임을 의미한다.
2. 합성 선호도 데이터는 저노이즈(low-noise), 고편향(high-bias)이다. 이는 AI 피드백 데이터로 시작하기가 더 쉽지만, 데이터에 체계적으로 나타나는 모델에 대한 까다롭고 의도치 않은 이차적 영향이 있을 수 있음을 의미한다.

이 책은 AI 선호도 데이터를 RLHF 워크플로에 대체하고 강력한 평가 점수를 달성하는 방법을 보여주는 많은 학술적 결과를 강조하지만 [@miranda2024hybrid], 더 넓은 산업 동향은 RLHF 문헌이 더 불투명한 모범 사례와 분리되어 있는 방식을 보여준다.
산업 전반에서, 인간 데이터는 종종 상당한 해자와 주요 기술 우위로 여겨진다.

## 헌법적 AI

Anthropic이 Claude 모델에 사용하는 헌법적 AI (CAI) 방법론은 RLHF 학습을 위한 합성 데이터의 가장 초기에 문서화된 대규모 사용이다.
헌법적 AI는 두 가지 방식으로 합성 데이터를 생성하는 것을 포함한다:

1. "답변이 폭력을 조장하는가" 또는 "답변이 진실한가"와 같은 원칙 집합을 따르도록 지시 조정된 데이터의 비판을 생성한다. 모델이 질문에 대한 답변을 생성할 때, 헌법의 원칙 목록에 대해 답변을 확인하고 시간이 지남에 따라 답변을 개선한다. 그런 다음, 모델은 이 결과 데이터셋으로 미세조정된다.
2. 언어 모델을 사용해 헌법의 무작위 원칙의 맥락을 주어 어떤 완성물이 더 나은지 답변하게 함으로써 쌍별 선호도 데이터를 생성한다 (원칙 안내 보상 모델에 관한 연구와 유사하게 [@sun2024salmon]). 그런 다음, RLHF는 합성 데이터로 평상시처럼 진행되며, 따라서 RLAIF라는 이름이 붙는다.

대체로, CAI는 위의 두 번째 부분인 선호도 데이터로 알려져 있지만, 지시 데이터를 위해 도입된 방법들은 후처리 학습 전반에 걸친 일반 데이터 필터링 및 합성 데이터 생성 방법에서 사용된다.

CAI는 다음과 같이 공식화될 수 있다.

인간이 작성한 원칙 집합인 *헌법(constitution)*을 사용함으로써, Bai et al. 2022는 별도의 LLM을 사용해 미세조정에 사용되는 인공적인 선호도와 지시 데이터를 생성한다 [@bai2022constitutional].
헌법 $\mathcal{C}$는 비판 단계에서 초점을 맞출 특정 측면을 나타내는 작성된 원칙들의 집합이다.
지시 데이터는 원칙 $c_i \in \mathcal{C}$를 반복적으로 샘플링하고 모델에게 프롬프트 $x$에 대한 최신 출력 $y^i$를 $c_i$에 맞게 수정하도록 요청함으로써 큐레이션된다.
이는 비판에 사용된 원칙 $\{c_{0}, c_{1}, \cdots, c_{n-1}\}$에서 일련의 지시 변형 $\{y^0, y^1, \cdots, y^n\}$을 생성한다.
최종 데이터 포인트는 어떤 $n$에 대해 프롬프트 $x$와 최종 완성물 $y^n$이다.

선호도 데이터는 $\mathcal{C}$의 원칙 하위 집합을 피드백 모델의 맥락으로 사용하는 유사하지만 더 간단한 방식으로 구성된다.
피드백 모델은 프롬프트 $x$, 원칙 집합 $\{c_0, \cdots, c_n\}$, 그리고 이전 RLHF 데이터셋의 답변 (A)와 (B)로 레이블링된 두 완성물 $y_0$와 $y_1$을 제공받는다.
새 데이터 포인트는 언어 모델이 어느 출력 (A) 또는 (B)가 더 높은 품질이면서 명시된 원칙에 더 잘 정렬되는지 선택하게 함으로써 생성된다.
초기 모델에서는 모델에게 `The answer is: `로 프롬프트를 주고 어느 로짓 (A) 또는 (B)가 더 높은 확률을 가지는지 확인함으로써 수행될 수 있었지만, 이제 더 일반적으로 모델이 추론을 설명한 다음 답을 선택하는 방식으로 처리된다 -- 일반적으로 생성형 보상 모델의 한 유형으로 불린다 [@mahan2024generative].

**추가 읽기.** 헌법적 AI의 많은 관련 연구 방향과 확장이 있지만, 그 중 명확한 RLHF 및 후처리 학습 레시피 개선으로 문서화된 것은 거의 없다.

- OpenAI는 Model Spec [@openai2024modelspec]을 공개했으며, 이는 모델의 의도된 행동을 서술하는 문서로, 모델이 문서를 직접 참조하는 정렬 방법을 탐색하고 있다고 밝혔다 (이는 CAI의 가까운 동료로 볼 수 있다). OpenAI는 계속해서 Deliberative Alignment [@guan2024deliberative]라는 방법으로 o1과 같은 추론 모델을 학습시켜 이러한 안전 또는 행동 정책을 참조하면서 모델을 정렬했다.
- Anthropic은 Claude가 사용하는 헌법을 업데이트하면서 [@Anthropic2023ClaudesConstitution] CAI를 계속 사용하며, 집단이 모델을 위한 원칙에 어떻게 수렴하는지 그리고 사람들이 스스로 원칙을 만들고 Anthropic과 공유해 모델을 학습시킬 때 모델 행동이 어떻게 변화하는지 실험하고 있다 [@ganguli2023].
- 오픈소스 커뮤니티는 오픈 데이터셋에 적용된 CAI의 복제본을 탐구했고 [@Huang2024cai] LM 간의 대화 데이터 생성 탐구에도 활용했다 [@lambert2024self].
- 다른 연구는 다른 최적화 방법을 사용하는 원칙 기반 선호도 또는 피드백을 사용했다.
Sun et al. 2023 [@sun2023principledriven]은 원칙을 보상 모델의 맥락으로 사용하며, Dromedary 모델 [@sun2024salmon]을 학습시키는 데 사용되었다.
Glaese et al. 2022 [@glaese2022improving]는 RLHF 과정에서 인간 판단의 정확성을 향상시키기 위해 원칙을 사용한다.
Liu et al. 2025 [@liu2025inference]는 추론 시에 자체 원칙을 생성하고 최종 점수를 제공하는 데 사용하는 보상 모델을 학습시킨다.
Franken et al. 2024 [@franken2024self]는 원칙 따르기를 사전 학습된 모델이 레이블 없이 학습할 수 있는 상호 정보 극대화 문제로 공식화한다.

## 판단을 위한 특정 LLM 구축

RLAIF 방법들이 더 보편화됨에 따라, 많은 사람들이 응답 생성에 사용하는 것과 동일한 모델을 비판이나 평가 생성에 사용해야 하는지 궁금해했다.
구체적으로, LLM-as-a-judge로 사용되는 LLM의 보정이 문제가 되고 있다.
여러 연구는 LLM이 일관성 없는 평가자라는 것을 보여주었으며 [@wang2023large], 다른 모델의 응답보다 자신의 응답을 선호한다(자기 선호 편향이라고 불림) [@panickssery2024llm].

이러한 편향의 결과로, 많은 사람들이 물었다: 이 레이블링 작업만을 위해 별도의 모델을 학습시키는 것이 해결책이 될 수 있는가?
Shepherd [@wang2023shepherd] 및 CriticLLM [@ke2023critiquellm]과 같은 비평 모델 또는 Auto-J [@li2023generative], Prometheus [@kim2023prometheus], Prometheus 2 [@kim2024prometheus], 또는 Prometheus-Vision [@lee2024prometheus]과 같은 응답 성능 평가 모델처럼 데이터 레이블링 도구로 프론티어 모델을 대체할 목적으로 여러 모델이 출시되었지만, 문서화된 학습 레시피에서 널리 채택되지는 않았다.
일부는 반복 샘플링을 통해 추론 확장 [@brown2024large] [@zhao2025sample] [@kalra2025verdict], 자기 개선 [@madaan2023self], 또는 토너먼트 랭킹 [@pace2024west]이 진정한 판단이나 더 높은 품질의 선호도 쌍의 더 나은 추정치를 제공한다고 본다.
다른 보정 기법들은 모델의 생성과 판단 능력을 함께 발전시킨다 [@wu2024meta].
편향이 존재하지만, 선도 언어 모델들은 이 작업을 위해 광범위하게 학습되었다는 것이 받아들여지고 있다 -- AI 연구소 내부 운영에도 필요하고 고객들이 광범위하게 사용하기 때문에 -- 따라서 작업에 공개 인터넷에 노출되지 않는 상당한 비공개 정보가 포함되지 않는 한, 일반적으로 자체 판사를 학습시킬 필요가 없다.

## 루브릭: 학습을 위한 프롬프트 특정 AI 피드백

AI 피드백의 학습에서의 역할은 2024년 말과 2025년으로 이어지면서 분야가 검증 가능한 보상을 활용한 강화학습 (RLVR)을 확장할 방법을 모색하면서 성장했다 (7장 참조).
루브릭(rubric)이라는 아이디어는 명확하게 검증 가능한 답이 없는 프롬프트에 대해 거의 검증 가능한 기준을 얻는 방법으로 등장했다.
이는 모델이 문제에 대해 여러 답변을 생성하고 최상의 답변을 향해 (RL로) 업데이트할 수 있도록 할 것이다.
이 아이디어는 이 장에서 논의된 다른 방법들과 밀접하게 관련되어 있으며, 아마도 LLM 판사와 합성 데이터 관행이 산업 전반에 걸쳐 개선되면서 작동하기 시작했을 것이다.
이제, 루브릭을 보상으로 사용하는 RL은 과학적 추론이나 사실성 [@gunjal2025rubrics; @viswanathan2025checklists; @rezaei2025onlinerubrics; @liu2025openrubrics]과 같은 기술에서 의미 있는 개선을 제공하는 것으로 확립되었다.

아래에 관련 프롬프트와 함께 루브릭 예시가 나와 있다 [@liu2025openrubrics]:
```text
**Prompt**: As a museum curator, can you suggest five obscure artifacts that would be perfect for a "Mysteries of the Ancient World" exhibit? Each artifact should come from a different culture and time period, with a brief description of their historical significance and mysterious origins. These artifacts should leave visitors wondering about the secrets and lost knowledge of our past. Thank you for your expertise in bringing this exhibit to life.

** Rubric**: 
1. The response includes exactly five distinct artifacts as requested. [Hard Rule] 
2. The response ensures each artifact originates from a different culture and time period. [Hard Rule] 
3. The response provides a brief description of each artifact's historical significance. [Hard Rule] 
4. The response provides a brief description of each artifact's mysterious origins or unexplained aspects. [Hard Rule] 
5. The response conveys a sense of intrigue and mystery that aligns with the theme of the exhibit. [Hard Rule] 
6. The response clearly and accurately communicates information in a well-organized and coherent manner. [Principle] 
7. The response demonstrates precision and clarity by avoiding unnecessary or irrelevant details. [Principle] 
8. The response uses informative and engaging language that stimulates curiosity and critical thinking. [Principle] 
9. The response shows thoughtful selection by ensuring each example contributes uniquely to the overall theme without redundancy. [Principle] 
10. The response maintains consistency in style and format to enhance readability and comprehension. [Principle]
```

`[Hard Rule]`과 `[Principle]`은 특정 피드백의 우선순위를 나타내는 특수 태그다. 중요도를 나타내는 다른 방법들도 사용될 수 있으며, 예를 들어 단순한 우선순위 번호가 있다.

루브릭 생성은 일반적으로 학습 데이터의 프롬프트별로 수행되며, 이는 준비 과정에서 상당한 합성 데이터 비용을 누적시킨다.
이를 완화하기 위해, 일반 루브릭이 종종 도메인별 시작점으로 적용되고, 그런 다음 프롬프트별 세밀한 루브릭 점수가 감독 언어 모델에 의해 학습을 위한 피드백을 안내하도록 할당된다.
과학 과제를 위한 루브릭 생성 예시 프롬프트가 아래에 나와 있다 [@gunjal2025rubrics]:

```text
You are an expert rubric writer for science questions in the domains of Biology, Physics, and Chemistry. 
Your job is to generate a self-contained set of evaluation criteria ("rubrics") for judging how good a response is to a given question in one of these domains. 
Rubrics can cover aspects such as factual correctness, depth of reasoning, clarity, completeness, style, helpfulness, and common pitfalls. 
Each rubric item must be fully self-contained so that non-expert readers need not consult
any external information.

Inputs:
- question: The full question text.
- reference_answer: The ideal answer, including any key facts or explanations.

Total items:
- Choose 7-20 rubric items based on question complexity.

Each rubric item must include exactly three keys:
1. title (2-4 words)
2. description: One sentence beginning with its category prefix, explicitly stating what to look for. 

For example:
- Essential Criteria: States that in the described closed system, the total mechanical energy (kinetic plus potential)
before the event equals the total mechanical energy after the event.
- Important Criteria: Breaks down numerical energy values for each stage, demonstrating that initial kinetic
energy plus initial potential energy equals final kinetic energy plus final potential energy.
- Optional Criteria: Provides a concrete example, such as a pendulum converting between kinetic and potential
energy, to illustrate how energy shifts within the system.
- Pitfall Criteria: Does not mention that frictional or air-resistance losses are assumed negligible when applying
conservation of mechanical energy.

3. weight: For Essential/Important/Optional, use 1-5 (5 = most important); for Pitfall, use -1 or -2.

Category guidance:
- Essential: Critical facts or safety checks; omission invalidates the response.
- Important: Key reasoning or completeness; strongly affects quality.
- Optional: Nice-to-have style or extra depth.
- Pitfall: Common mistakes or omissions; highlight things often missed.

Format notes:
- When referring to answer choices, explicitly say "Identifies (A)", "Identifies (B)", etc.
- If a clear conclusion is required (e.g. "The final answer is (B)"), include an Essential Criteria for it.
- If reasoning should precede the final answer, include an Important Criteria to that effect.
- If brevity is valued, include an Optional Criteria about conciseness.

Output: Provide a JSON array of rubric objects. Each object must contain exactly three keys-title, description, and weight.
Do not copy large blocks of the question or reference_answer into the text. Each description must begin with its category
prefix, and no extra keys are allowed.
Now, given the question and reference_answer, generate the rubric as described. 
The reference answer is an ideal response but not necessarily exhaustive; use it only as guidance.
```

또 다른 더 간단한 예시는 다음과 같다 [@rezaei2025onlinerubrics]:

```text
SYSTEM:
You generate evaluation rubrics for grading an assistant's response to a user prompt.

Rubric design rules:
- Each criterion must be atomic (one thing), objective as possible, and written so a grader can apply it consistently.
- Avoid redundant/overlapping criteria; prefer criteria that partition different failure modes.
- Make criteria self-contained (don't rely on unstated context).
- Include an importance weight for each criterion.

Output format (JSON only):
{
  "initial_reasoning": "<brief rationale for what matters for this prompt>",
  "rubrics": [
    {
      "reasoning": "<why this criterion matters>",
      "criterion": "<clear, testable criterion>",
      "weight": <integer 1-10>
    },
    ...
  ]
}

USER:
User prompt:
{prompt}

Generate the rubric JSON now.
```

보다시피, 프롬프트들은 매우 상세할 수 있으며 학습 설정에 맞게 조율된다.

RL 학습과 함께하는 루브릭은 지시 따르기 [@he2025advancedif], 심층 연구 [@shao2025drtulu], 심층 연구 에이전트 평가 [@sharma2025researchrubrics], 또는 장문 생성 [@ruan2025expertlongbench]에 대한 초기 응용을 넘어 계속 발전할 것이다.
