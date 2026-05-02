<!--
  Copyright (c) 2025-2026 Nathan Lambert.
  Licensed under CC BY-NC-SA 4.0:
  https://creativecommons.org/licenses/by-nc-sa/4.0/
  Full license: https://github.com/natolambert/rlhf-book/blob/main/LICENSE-CHAPTERS
-->
---
prev-chapter: "정규화"
prev-url: "15-regularization"
page-title: 평가
search-title: "16장: 평가"
next-chapter: "모델 캐릭터와 제품 구성"
next-url: "17-product"
---

# 평가

평가 (Evaluation)는 이 책에서 상세히 다루는 학습 과정의 품질과 영향을 이해하기 위해 사용되는 기법들의 집합이다.
평가는 일반적으로 벤치마크 (benchmark, 대표적인 예로 MMLU, GPQA, SWE-Bench, MATH 등이 있다)를 통해 표현되며, 벤치마크는 모델의 특정 속성을 측정하기 위해 설계된 질문이나 환경의 이산 집합이다.
평가는 끊임없이 진화하는 접근 방식이므로, 여기서는 RLHF 내 평가의 최근 흐름과 언어 모델링의 미래에도 이어질 공통 주제들을 소개한다.
언어 모델 평가, 특히 후처리 학습 (post-training)에서의 평가를 이해하는 핵심은, 현재의 주요 평가 체계가 인기 있는 학습 모범 사례 및 목표를 반영한다는 점이다.
어려운 평가들이 언어 모델을 새로운 영역으로 이끌어 나가는 한편, 대부분의 평가는 새 모델에 유용한 신호를 제공하는 방향으로 설계된다.

여러 면에서, 이 장은 RLHF 초기 역사에서 나타난 주요 평가 체계들의 단면을 소개하여 독자들이 공통 주제, 세부 사항, 그리고 실패 원인들을 이해할 수 있도록 구성되어 있다.

RLHF와 후처리 학습의 평가는 초기 역사에서 몇 가지 뚜렷한 단계를 거쳐 왔다:

1. **초기 대화 단계**: RLHF 또는 선호도 조정 (preference tuning)으로 학습된 초기 모델들은 GPT-4 같은 강력한 모델 대비 대화 성능을 측정하는 평가에 집중했다. 초기 사례로는 MT-Bench [@zheng2023judging], AlpacaEval [@dubois2024length], Arena-Hard [@li2024crowdsourced] 등이 있다. 이 벤치마크들은 GPT-4 같은 모델을 사용해 응답을 채점하는 LLM-as-a-judge 방식으로 인간 평가자를 대체했으며, 이는 인간 평가 기준을 비용 효율적으로 확장하는 방법이었다 (12장 참조). 모델은 좁은 범위에서 평가되었으며, 이는 현재 "대화" 또는 "지시 따르기" 영역으로 분류된다.
2. **다중 기술 시대**: 시간이 지나면서 RLHF가 단순 대화 이상의 더 많은 기술을 향상시키는 데 사용될 수 있다는 것이 일반적인 관행으로 자리 잡았다. 예를 들어, Tülu 평가 모음은 지식 (MMLU [@hendrycks2020measuring], PopQA [@mallen2023llm_memorization], TruthfulQA [@lin2021truthfulqa]), 추론 (BigBenchHard [@suzgun2022challenging], DROP [@dua2019drop]), 수학 (MATH [@hendrycksmath2021], GSM8K [@cobbe2021gsm8k]), 코딩 (HumanEval [@chen2021codex], HumanEval+ [@evalplus]), 지시 따르기 [@zhou2023instructionfollowingevaluationlargelanguage], 그리고 안전성 (다양한 평가들의 복합) 과제를 포함했다. 이는 후처리 학습이 안전성과 대화를 넘어 다면적인 해결책으로 자리 잡은 영역을 반영한다.
3. **추론 및 도구 사용**: 현재 후처리 학습 시대는 어려운 추론과 도구 사용 문제에 초점을 맞추고 있다. 여기에는 GPQA Diamond [@rein2023gpqa]와 Humanity's Last Exam [@phan2025hle] 같은 지식 집약적 과제, SWE-Bench+ [@aleithan2024swebenchplus]와 LiveCodeBench [@jain2024livecodebench] 같은 복잡한 소프트웨어 엔지니어링 과제, 그리고 최근 AIME 대회로 대표되는 도전적인 수학 문제들이 포함된다.

이 밖에도 새로운 영역은 계속 진화할 것이다.
AI가 점점 더 산업화된 분야가 됨에 따라, 평가의 인센티브가 변화하고 다중 이해관계자적 성격을 띠게 되고 있다.
ChatGPT 출시 이후, Scale 리더보드 [@scale2024seal] 같은 민간 평가, Arena [@chiang2024chatbot] 같은 커뮤니티 주도 평가, 그리고 ArtificialAnalysis와 Epoch AI 같은 제3자 평가 기관들이 급증했다.
이 장 전반에 걸쳐 이러한 평가들이 어떻게 구현되고 이해되었는지에 대한 세부 사항을 다룰 것이다.

## 프롬프트 형식: 퓨샷에서 제로샷, 그리고 사고의 연쇄까지

언어 모델에 **프롬프트 (prompting)**를 입력하는 것은 그 자체로는 단순하고 꽤 자연스러운 행위이지만, 동시에 연습하고 정교하게 다듬을 수 있는 기술 또는 예술로 여겨지기도 한다 [@schulhoff2024prompt].
프롬프트는 언어 모델에 정보와 맥락을 구조화하는 방식이다.
일반적인 상호작용에서 프롬프트는 비교적 단순하다.
고급 시나리오에서는 잘 만들어진 프롬프트가 특정 일회성 사용 사례의 성공과 실패를 가른다.

평가에 있어서 프롬프트 기법은 모델 성능에 상당한 영향을 미칠 수 있다.
일부 프롬프트 기법, 예를 들어 아래에서 논의할 형식화는 모델 성능을 60%에서 거의 0으로 떨어뜨릴 수 있다.
마찬가지로 프롬프트 변경은 학습 중 모델이 더 잘 배우는 데 도움이 될 수 있다.
통상적으로, 모델에 좋은 프롬프트를 입력하면 미래 모델을 사용하는 것 같은 주관적 경험을 줄 수 있으며, 일반적인 사용 범위를 벗어난 성능을 끌어낼 수 있다.

프롬프팅을 통한 이득은 일반적으로 데이터 개선이나 학습 알고리즘 향상 같은 핵심 영역보다 작지만, 최종 제품에서는 상당한 차이를 만들 수 있다.
더 큰 시사점은, 강력하고 선도적인 모델을 학습시킬 때 성능을 조금 더 끌어올리기보다는 성능을 무너뜨리고 급락시키기가 더 쉽다는 것이다.

현대 언어 모델에 효과적으로 프롬프팅하면 모델이 응답할 전체 보고서를 준비하는 것을 포함할 수 있다 (종종 수천 개의 생성된 토큰 포함).
이 동작은 언어 모델 성능이 측정되고 이해되는 방식의 많은 변화에서 비롯된 것이다.

초기 언어 모델들은 지능적인 자동 완성으로만 사용되었다.
이 모델들을 더 개방적인 방식으로 사용하기 위해, 여러 예시들을 모델에 보여준 후 불완전한 구문 형태의 프롬프트를 입력했다. 이것이 퓨샷 (few-shot) 또는 문맥 내 학습 (ICL, in-context learning) [@brown2020language]이라 불리며, 당시에는 지시 조정이나 RLHF가 관여하지 않았다.
일반적인 평가의 경우, 이것은 다음과 같은 형태가 된다:

```text
# Few-Shot Prompt for a Question-Answering Task
You are a helpful assistant. Below are example interactions to guide your style:

### Example 1
User: "What is the capital of France?"
Assistant: "The capital of France is Paris."

### Example 2
User: "Who wrote the novel '1984'?"
Assistant: "George Orwell wrote '1984.'"

# Now continue the conversation using the same style.
User: "Can you explain what a neural network is?"
Assistant:
```

여기서 답변을 평가하는 방법은 여러 가지가 있다. MMLU 스타일의 질문처럼 모델이 여러 답변 중 하나를 선택해야 하는 경우를 고려하면:

```text
# Few-Shot Prompt

Below are examples of MMLU-style questions and answers:

### Example 1
Q: A right triangle has legs of lengths 3 and 4. What is the length of its hypotenuse?
Choices:
(A) 5
(B) 6
(C) 7
(D) 8

Correct Answer: (A)

### Example 2
Q: Which of the following is the chemical symbol for Sodium?
Choices:
(A) Na
(B) S
(C) N
(D) Ca

Correct Answer: (A)

### Now answer the new question in the same style:

Q: Which theorem states that if a function f is continuous on a closed interval [a,b], then f must attain both a maximum and a minimum on that interval?
Choices:
(A) The Mean Value Theorem
(B) The Intermediate Value Theorem
(C) The Extreme Value Theorem
(D) Rolle's Theorem

Correct Answer:
```

언어 모델이 여기서 답변을 제공하도록 하려면, 일부 샘플링 파라미터를 기반으로 토큰을 생성하여 답이 A, B, C, D 중 맞는지 확인하거나 (위와 같은 형식은 [@robinson2023leveraging]에서 제안됨), 각 토큰의 로그 확률 (log-probability)을 보고 정답이 더 높은 확률을 가진다면 정답으로 채점하는 방법을 사용할 수 있다.

이러한 평가 세부 사항을 잠시 살펴보자.
전자는 단일 시도에 대한 정확 일치 (exact match)나 여러 샘플을 집계할 때의 다수결 투표 (majority voting)라 하고 (pass@k는 기능적 정확성을 테스트하는 코딩 평가의 유사한 지표이다), 후자는 (조건부) 로그 우도 (log-likelihood) 채점이라 하며, 조건은 프롬프트이다.
핵심 차이는 기저 확률 분포에서의 샘플링은 자연적으로 무작위성을 추가하는 반면, 모델이 토큰에 대해 출력하는 로그 확률은 (미소한 수치적 차이를 무시할 때) 정적이라는 것이다.

로그 우도 채점에는 두 가지 잠재적 구현이 있다 -- 첫째로 글자 (A) 의 확률이나 "The Mean Value Theorem"이라는 답변의 확률을 볼 수 있다.
두 방법 모두 허용 가능한 지표이지만, 글자 답을 예측하는 것이 잠재적으로 여러 토큰에 걸친 완전한 답변 확률보다 훨씬 단순하다.
로그 우도 채점은 정확 일치에 필요한 질문-답변 형식이 부족한 사전 학습 (pretraining) 평가에서 더 일반적이며, 정확 일치는 후처리 학습에서 표준이다 [@teamolmo2025olmo3].

정확 일치에는 다른 문제들이 있는데, 예를 들어 엄격한 형식 접미사가 필요하거나 (예: `The answer is:`) 생성된 텍스트에서 어디서든 답을 감지하기 위해 정규 표현식을 사용해야 한다 (예: `(C)` 또는 답변 문자열 자체 탐색).
평가 형식이 모델의 생성 방식과 일치하지 않으면 점수가 급락할 수 있다.
언어 모델 평가는 형식화가 병목이 되지 않아 모델의 완전한 능력을 테스트할 수 있을 때 가장 잘 이루어진다.
형식에 구애받지 않는 평가를 달성하는 데는 상당한 노력과 조정이 필요하며, 실제로는 드물다.

평가의 역사로 돌아가서.
위에서 사용된 설정에 관계없이, 퓨샷 프롬프팅의 일반적인 과제는 모델이 형식을 따르지 않는다는 것이며, 이는 오답으로 채점된다.
평가 도메인을 설계할 때, 문맥 내에서 사용되는 예시 수는 종종 설계 파라미터로 간주되며 3개에서 8개 또는 그 이상의 범위를 갖는다.

퓨샷 프롬프팅의 발전 과정에서 모델이 따를 수 있는 사고의 연쇄 (CoT, chain-of-thought) 예시를 포함하는 아이디어가 생겨났다.
이는 문맥 내 예시에 작성된 추론이 포함된 형태로 나타나며, 아래와 같은 모습이다 (이후 추론 단계를 생성하도록 명시적으로 프롬프팅하는 방식으로 대체됨) [@wei2022chain]:

```text
# standard prompting
Q: Roger has 5 tennis balls. He buys 2 more cans of tennis balls. Each can has 3 tennis balls. How many tennis balls does he have now?

A: The answer is 11.

Q: The cafeteria had 23 apples. If they used 20 to make lunch and bought 6 more, how many apples do they have?

A: The answer is ...

# chain-of-thought prompting
Q: Roger has 5 tennis balls. He buys 2 more cans of tennis balls. Each can has 3 tennis balls. How many tennis balls does he have now?

A: Roger started with 5 balls. 2 cans of 3 tennis balls each is 6 tennis balls. 5 + 6 = 11. The answer is 11.

Q: The cafeteria had 23 apples. If they used 20 to make lunch and bought 6 more, how many apples do they have?

A: The cafeteria had 23 apples originally. They..
```

시간이 지나면서 언어 모델들이 강해짐에 따라 제로샷 (zero-shot) 평가, 즉 "제로샷 학습자 (zero-shot learners)" [@wei2021finetuned]로 발전했다.
파인튜닝된 언어 네트워크 (FLAN, Finetuned Language Net)는 현대 지시 조정의 전신으로서 특정 과제에 미세조정된 언어 모델이 학습하지 않은 제로샷 질문에도 일반화할 수 있음을 보였다 [@wei2021finetuned] (유사한 결과가 T0 [@sanh2021multitask]에서도 발견됨).
이것이 지시 미세조정 (IFT, instruction fine-tuning)의 등장으로, RLHF와 후처리 학습의 중요한 선구자이다.
제로샷 질문은 다음과 같은 형태가 된다:

```text
User: "What is the capital of France?"
Assistant:
```

2022년부터 시작하는 이 시기에, InstructGPT 같은 초기 RLHF 핵심 연구들이 포함되기 시작했다.
이 모델들과 함께 이루어진 핵심 역량 및 사용 사례의 전환은 더욱 개방적인 사용이다.
더 개방적인 사용과 함께, 모델에서의 샘플링을 통한 평가가 실제 사용을 반영하므로 점점 인기를 끌게 되었다 -- 기술적으로 이것은 생성 기반 (정확 일치) 평가라고 할 수 있지만, 명확한 표준 용어는 없다.
ChatGPT 이후 최근 몇 년간 이 기간 동안, 일부 객관식 평가들은 여전히 RLHF 연구에서 사용되었는데, 이는 어떤 관행의 전환이든 일반적으로 수개월에서 수년이 걸리기 때문이다 (예를 들어, 이런 유형의 평가는 온도를 0으로 설정하고 A, B, C, D 문자를 샘플링하여 수행한다).

2024년 말과 2025년 초에 추론 모델의 부상과 함께, 모델 동작의 중요한 변화는 모든 답변 전에 긴 사고의 연쇄 (CoT) 추론 과정이 추가된 것이다.
이 모델들은 더 이상 [@kojima2022large]에서 제안된 표준 구문인 "단계별로 생각해 보세요"라고 프롬프팅할 필요가 없었다.
다음 평가 관행의 진화는 사고의 연쇄 추론이 포함된 생성 기반 (정확 일치) 평가이다 (따라서 최상의 성능을 위해 거의 항상 0 이상의 온도가 필요하다).

예를 들어, 일부 설정에서는 모든 질문이나 카테고리에 대해 모델의 동작을 추출하는 데 도움이 되는 특별히 설계된 프롬프트가 있다.
Tülu 3는 객관식 질문에 대한 CoT 답변에 사용된 일부 프롬프트를 자세히 설명한 초기 중요 논문이다 [@lambert2024t].
아래는 MMLU에 사용된 예시 프롬프트로, 단일 토큰 답변 샘플링에서 정확 일치 답변 확인을 포함한 긴 형식의 CoT로 전환된 평가 중 하나이다.

```text
Answer the following multiple-choice question by giving the correct answer letter in parentheses.
Provide CONCISE reasoning for the answer, and make sure to finish the response with "Therefore, the answer is (ANSWER_LETTER)" where (ANSWER_LETTER) is one of (A), (B), (C), (D), (E), etc.

Question: {question}
(A) {choice_A}
(B) {choice_B}
(C) ...

Answer the above question and REMEMBER to finish your response with the exact phrase "Therefore, the answer is (ANSWER_LETTER)" where (ANSWER_LETTER) is one of (A), (B), (C), (D), (E), etc.
```

이것은, 특히 모델들이 생각 토큰과 답변 토큰을 구분하는 특수 형식을 사용할 때, 평가 체계에 가장 최근의 주요 업데이트를 필요로 하게 만들었다.
평가는 사고의 연쇄 프롬프팅을 통해 생성적인 방식으로 응답하도록 모델들을 테스트하는 방향으로 나아가고 있다.

## 외부 평가 비교가 신뢰하기 어려운 이유

AI 기업들의 모델 발표에서 언어 모델 평가들은 큰 오차 범위를 갖는 다른 보도자료들과만 비교될 수 있다 -- 즉, 약간 더 좋거나 나쁜 모델은 동등한 것으로 간주되어야 한다 -- 왜냐하면 각 기업이 내부적으로 평가를 위해 사용하는 절차가 모델 간에 통제되지 않고 명시적으로 문서화되지 않기 때문이다.
예를 들어, Olmo 3 프로젝트에서 저자들은 추론 모델 시대의 대부분의 후처리 학습 평가가 평가 설정이 일정하게 유지될 때 표준 편차가 0.25에서 1.5 포인트 사이임을 발견했다 [@teamolmo2025olmo3] -- 더 큰 점수 변동은 다른 프롬프트나 샘플링 파라미터를 사용하는 것에서 올 수 있다.
연구소들은 모델을 더 유용하게 만들기 위해 학습 중에 평가에서 점진적으로 더 나은 성능을 목표로 하는데 (힐클라이밍), 전통적으로 학습 세트, 개발 (검증 세트라고도 함) 세트, 보류된 평가 세트 (테스트 세트라고도 함)의 혼합을 사용한다.
힐클라이밍 (hillclimbing)은 일련의 목표 벤치마크에서 모델을 점진적으로 향상시키는 관행을 설명하는 구어체 용어이다.
커뮤니티가 선도 모델을 비교하는 데 사용하는 공개 평가의 경우, 어떤 것들이 학습에 사용되었고 어떤 것들이 테스트를 위해 보류되었는지 알 수 없다.

평가 점수가 기업 마케팅 계획의 핵심 요소가 됨에 따라, 기업 내부에서의 구현이 변화했다.
GSM8k나 MATH 같은 중요한 평가에 주요 AI 연구소들이 "커스텀 프롬프트"를 사용한다는 소문이 있다.
이러한 관행은 빠르게 진화한다.

언어 모델 평가 스택이 마케팅으로 인식되는 이유는 평가에 확고한 진실의 원천이 없기 때문이다.
프런티어 연구소 내부에서 일어나고 있는 일은 평가 모음이 내부 필요에 맞게 조정되고 있다는 것이다.
결과가 공유될 때, 우리는 연구소가 자신들의 모델에서 얻은 숫자 형태의 출력을 얻지만, 그 함수에 대한 모든 입력을 얻지는 못한다.
입력들은 매우 민감한 구성 요소들이며, OpenAI, Meta, Anthropic, Google 모두에서 다르다.
완전히 공개된 평가 표준이더라도 재현성을 보장하기 어렵다.
자신의 모델에 노력을 집중하는 것만이 반복 가능한 평가 기법에 근접할 수 있는 유일한 방법이다.
기술 팀에서 시작하는 마케팅에는 좋은 의도가 깔려 있다.

여러 연구소에서 평가를 비교할 때 혼란의 또 다른 예는 평가 비교에 추론 시간 스케일링을 추가하는 것이다.
추론 시간 스케일링은 모델이 추론 시에 더 많은 토큰을 사용함으로써 성능을 향상시킬 수 있음을 보여준다.
따라서 추론에 사용되는 총 토큰 수에 따라 평가 점수를 제어하는 것이 중요하지만, 아직 일반적인 관행이 아니다.

후처리 학습에서 데이터 형식이 어떻게 구성되느냐에 따라 모델은 평가 형식에 걸쳐 상당한 차이를 보일 것이다.
예를 들어, 두 개의 인기 있는 공개 수학 데이터셋인 NuminaMath [@li2024numinamath]와 MetaMath [@yu2023metamath]는 답변 형식이 약간 다른 방식으로 충돌한다 -- Numina는 답을 `\boxed{XYZ}`에 넣고 MetaMath는 답을 `The answer is: XYZ` 뒤에 넣는다 -- 두 데이터셋으로 학습하면 어느 하나로만 학습하는 것보다 성능이 떨어질 수 있다.
강력한 모델들은 여러 형식에서 기능할 수 있도록 학습되지만, 일반적으로 가장 강한 형식이 있다.

결국 우리는 폐쇄 모델 평가의 현황에 관한 몇 가지 핵심 사항이 남는다:

- 우리는 연구소들이 올라타고 있는 핵심 테스트 세트를 알거나 반드시 갖고 있지 않으므로, 일부 평가는 대리 지표이다.
- 프런티어 모델의 추론은 특수 시스템 프롬프트, 특수 토큰 등으로 점점 복잡해지고 있으며, 그것이 평가에 어떤 영향을 미치는지 모른다.
- 폐쇄 평가를 수치로 보고하는 데 사용되는 모든 형식과 세부 사항을 알지 못한다.

이러한 모든 역학 관계와 지난 몇 년간 AI 모델의 매우 빠른 발전은 @fig:benchmark-saturation 의 것과 유사한 유명한 도표를 만들어 내는데, 각 시대의 유행하는 벤치마크들이 매우 빠르게 해결된다.
각 벤치마크 수준에서 이 역학을 설명하는 일반적인 용어는 포화 (saturation)이다.
각 벤치마크가 100%에 가까워짐에 따라, 더 어려운 (또는 많은 경우 잘못 레이블된) 데이터 포인트만 남아 있어 모델의 진행 상황의 측정 (또는 두 모델 간의 비교)으로서 덜 신뢰할 수 있게 된다.

![주요 AI 평가들이 시간이 지남에 따라 빠르게 포화되는 방식을 보여주는 Epoch AI의 보고서 (포화는 주어진 벤치마크가 완전한 성능에 도달하여 모델이 더 이상 의미 있는 신호를 갖지 못하는 시점이다). 라이선스 CC-BY.](images/benchmark-performance.jpeg){#fig:benchmark-saturation}

## 연구소들이 실제로 내부적으로 평가를 사용하여 모델을 개선하는 방법

프런티어 언어 모델의 평가는 오늘날 과학만큼이나 예술이기도 하며, 다른 그룹들이 최첨단 언어 모델을 이해하기 위해 평가를 정확히 어떻게 사용하는지 규정하는 것은 그 자체로 교과서가 될 것이다.

다른 그룹들은 독립성을 유지하기 위해 다른 평가들을 선택한다, 즉 그것들을 진정한 테스트 세트로 만들지만, 아무도 어떤 것을 선택하는지 공개하지 않는다.
예를 들어, 인기 있는 추론 평가인 MATH와 GSM8k 둘 다 성능을 쉽게 향상시키는 데 사용될 수 있는 프롬프트가 있는 학습 세트를 가지고 있다.
같은 분포의 프롬프트로 성능을 향상시키는 것은 일반 수학 데이터로 학습하여 이러한 과제에 일반화하는 것과는 매우 다르다.

실제로, 이러한 *학습 세트들*은 매우 높은 품질의 데이터를 포함하고 있어 모델들이 그것으로 학습하면 이익을 얻을 것이다.
이 회사들이 해당 평가를 추적할 핵심 지표로 사용하지 *않는다면*, 높은 품질의 데이터가 모델 개발의 주요 제한 요소이므로 평가 세트로 학습하는 것이 실용적인 결정일 수 있다.

선도 AI 연구소들은 몇 가지 핵심 평가에 집중하여 힐클라이밍을 하고, 마지막에 핵심 공개 세트에 대한 점수를 보고한다.
핵심 포인트는 GPT-4 보고서 [@achiam2023gpt]의 스케일링에서 교차 엔트로피 손실 예측을 위한 데이터셋과 같이, 진행 상황을 추적하기 위한 일부 평가들이 종종 공개되지 않는다는 것이다.

후처리 학습 평가는 인간 평가에 크게 의존한다.
생성 언어 모델에 대한 인간 평가는 엘로 (Elo) 순위를 산출하고 (헌법적 AI 같은 초기 Anthropic 논문에서 인기가 있었다), 보상 모델에 대한 인간 평가는 일치도를 보여준다.
이것들은 A/B 테스트 창을 통해 두 개의 다른 모델을 사용자에게 서비스하는 방식으로도 얻을 수 있다 ([선호도 데이터에 관한 장](https://rlhfbook.com/c/11-preference-data) 참조).

그들이 집중하기로 선택한 제한된 평가 세트는 평가와 학습 사이의 긴밀한 연결을 형성한다.
한때 집중된 평가 중 하나는 MMLU였다.
GPQA는 과학적 역량에 대한 커뮤니티 집중 증가로 인해 추론 모델 등장 동안 매우 인기 있었다.
연구소들은 자신들의 필요에 더 적합하게 만들기 위해 평가들을 변경할 것이며, 예를 들어 OpenAI는 SWE-Bench-Verified [@openai2024swebench]를 출시했다.
공개적으로 접근할 수 없는 각 프런티어 연구소가 구축하거나 구매한 내부 평가들이 훨씬 더 많다.

내부적으로 평가를 개선하는 것이 후속 학습에 갖는 핵심 역량은 **학습 실행을 비교할 때 통계적 검정력을 향상시키는 것**이다.
평가를 변경함으로써, 이 연구소들은 더 많은 정보에 입각한 학습 결정을 내리기 위해 우선화된 신호의 노이즈를 줄인다.

이것은 현대 언어 모델 학습 스택에서 후처리 학습의 정교함에 의해 더욱 복잡해진다.
오늘날 언어 모델 평가에는 (답변의 로그 확률만 보는 것이 아닌) 상당한 양의 토큰 생성이 포함되므로 컴퓨팅 비용이 든다.
프런티어 연구소들이 많은 과제에서 성능을 높이기 위해 작은 트릭들을 사용한다는 것은 당연하게 받아들여지는데, 가장 일반적인 설명은 특정 평가를 위한 일회성 프롬프트이다.

## 오염

현재 언어 모델 관행 (즉, RLHF와 후처리 학습에만 국한되지 않음)의 주요 문제는 학습에서 평가 데이터셋의 데이터를 의도적 또는 비의도적으로 사용하는 것이다.
이것을 *데이터셋 오염 (dataset contamination)* (*데이터 누출 (data leakage)*의 한 형태)이라 하며, 이를 방지하는 관행을 각각 *오염 제거 (decontamination)*라 한다.
데이터셋을 오염 제거하기 위해, n-그램 단어/서브워드 토큰 겹침이나 고정 길이 문자 부분문자열 일치 (예: 50자) [@singh2024evaluation]를 찾아 학습 및 테스트 데이터셋에 대한 검색을 수행한다.
데이터가 오염되는 방법은 많지만, 가장 일반적인 것은 웹에서 여러 단계를 위한 학습 데이터를 스크래핑하는 것이다.
벤치마크들은 종종 크롤링되는 공개 웹 도메인에 나열되거나, 사용자들이 모델에 질문을 입력하면 향후 모델을 위한 후보 학습 데이터에 포함될 수 있다.

예를 들어, Tülu 3의 평가 모음 오염 제거 과정에서, 저자들은 인기 있는 공개 데이터셋이 RLHF를 위한 인기 있는 평가들로 오염되었다는 것을 발견했다 [@lambert2024t].
이러한 겹침에는 UltraFeedback의 TruthfulQA 오염, Evol-CodeAlpaca의 HumanEval 오염, NuminaMath의 MATH 오염, WildChat의 안전성 평가 오염이 포함된다.
이것들은 학습 프롬프트에서 평가 세트의 정확한 프롬프트까지 8-그램 겹침을 통해 발견되었다.

다른 경우에는, 수학 문제의 단어는 같게 유지하고 숫자만 변경하는 것처럼 벤치마크에 매우 근접한 데이터로 학습된 것으로 발견된 모델들이 있으며, 이는 후처리 학습 체계에서 특이한 동작을 초래할 수 있다. 예를 들어, 랜덤 보상으로 RL 학습 시 벤치마크가 향상되는데, 이는 모델이 특정 유형의 데이터 오염이 있는 경우에만 성능이 향상되어야 하는 인위적인 설정이다.
이러한 종류의 기반 모델 오염은, 모델이 특정 방식으로 동작하는 이유를 정확히 증명할 수 없는 경우, Qwen 2.5 및 Qwen 3 기반 모델 위에서의 많은 초기 RLVR 연구에서 상당한 교란 변수였다 [@shao2025spurious] [@wu2025reasoning].

훈련 데이터를 공개하거나 공개하지 않는 모델의 오염을 이해하기 위해, 원래 버전보다 약간 변경된 질문으로 벤치마크의 새 버전이 만들어져 (예: MATH [@huang2025math]), 어떤 모델이 원래 형식이나 질문을 맞추도록 학습되었는지 확인한다.
이러한 변형 벤치마크에서의 높은 분산은 오염의 확인이 아니며, 이를 증명하기 어렵다. 오히려, 실제 세계 성능으로 전환되지 않을 수 있는 특정 형식을 염두에 두고 학습된 모델을 나타낼 수 있다.


## 도구

사람들이 선택할 수 있는 많은 오픈소스 평가 도구들이 있다.
일부 도구들은 다음과 같다:

- 영국 안전 연구소의 Inspect AI [@inspectAI2024],
- Open LLM 리더보드 [@open-llm-leaderboard-v2]를 구동한 HuggingFace의 LightEval [@fourrier2023lighteval],
- 그들의 GPT-Neo-X 모델 인프라 위에 구축된 Eleuther AI의 평가 하네스 [@gao2023evalharness] (좋은 GPT-3 시대 평가 설정과 구성을 포함함) [@gpt-neox-20b],
- OLMES [@gu2024olmes]를 기반으로 한 AI2의 라이브러리,
- Stanford의 기반 모델 연구 센터의 HELM [@liang2023helm],
- Mosaic (현재 Databricks)의 Eval Gauntlet [@mosaicml2024gauntlet] 등이 있다.
