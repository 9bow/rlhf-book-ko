<!--
  Copyright (c) 2025-2026 Nathan Lambert.
  Licensed under CC BY-NC-SA 4.0:
  https://creativecommons.org/licenses/by-nc-sa/4.0/
  Full license: https://github.com/natolambert/rlhf-book/blob/main/LICENSE-CHAPTERS
-->
---
prev-chapter: "정의"
prev-url: "appendix-a-definitions"
page-title: "부록 B: \"단순한 스타일\"을 넘어서"
search-title: "부록 B: \"단순한 스타일\"을 넘어서"
meta-description: "RLHF와 사후 학습이 언어 모델에서 형성하는 스타일, 보정, 정보 제시 문제를 설명합니다."
next-chapter: "실용적 이슈"
next-url: "appendix-c-practical"
---

# "단순한 스타일"을 넘어서

RLHF의 초기 발전은 RLHF에 "단순한 스타일 변환"이라는 평판을 부여하거나, RLHF가 출력에서 정보를 제시하는 방식을 조작하는 방법에 대한 다른 혹독한 비판들을 만들어냈다.
이 부록은 스타일이 RLHF가 제공하는 가치를 이해하는 데 핵심인 이유와, 그것이 모델 역량과 사용자 경험 모두에 긍정적으로 영향을 미치는 이유를 설명한다.

RLHF가 단순히 스타일 변환에 관한 것이라는 생각이 RLHF 서사를 두 가지 이유로 가로막아 왔다.
첫 번째는 RLHF가 모델에 대한 작고 중요하지 않은 변화와 어떻게 연관되었는가 하는 것이다.
사람들이 스타일 변환에 대해 이야기할 때, 그들은 이것을 중요하거나 흥미로운 것으로 설명하지 않는다 -- 그들은 그것을 피상적인 것으로 생각한다.
그러나 스타일은 결코 끝나지 않는 인간적 가치의 원천이다; 이야기를 재전달하면 새로운 베스트셀러 책이 탄생할 수 있으며 ([사피엔스](https://en.wikipedia.org/wiki/Sapiens:_A_Brief_History_of_Humankind) 같은), 이것은 우리의 지적 생태계를 계속 발전시키는 근본적인 부분이다.
스타일은 정보가 무엇인지와 얽혀있다.

두 번째 이유는 많은 사람들이 잘 수행된 RLHF가 인기 있는 LLM 평가에서 점수를 높인다는 사실을 놓쳤다는 것이다.
Llama 3 [@dubey2024llama]에서 서로 다른 스타일이 실제로 평가를 의미 있게 향상시킬 수 있는 방법을 보았다.
Llama 3 Instruct 모델들은 Arena에서 극히 높은 점수를 받았으며, 더 재미있는 개성을 가졌기 때문이라고 받아들여진다 -- 그들은 같은 시대의 다른 모델들보다 더 간결하고 영리했다.
많은 LLM 사용자들이 집착하는 벤치마크 점수와 관계없이, RLHF가 언어 모델을 단순히 더 재미있게 만들 것이라면, 그것은 제공된 가치이다.

이 장 전반에 걸쳐, "수다스러움 (chattiness)"이라는 용어는 RLHF로 학습된 모델에서 응답 길이가 늘어나는 것을 포괄하는 데 사용되지만, 과도한 마크다운 사용, 이모티콘, 그리고 글머리 기호 목록으로 답변을 형식화하는 기법들도 포함한다.
이 장의 논의를 구체화하기 위해, 동일한 Tülu 3 계열의 두 모델 -- 지시 미세조정 (IFT)만 적용된 것과 길이 제어 DPO를 통해 선호도 미세조정 (PreFT)이 적용된 것 -- 의 예시 완성이 아래에 나와 있다.
먼저, `강화학습에서 인간 피드백이란 무엇인가?`라는 프롬프트에 대한 지시 미세조정된 완성 (정확한 모델은 [Llama 3.1 Tulu 3 70B SFT](https://huggingface.co/allenai/Llama-3.1-Tulu-3-70B-SFT)) [@lambert2024t]:

```text
Reinforcement learning from human feedback, also known as human-in-the-loop reinforcement learning, is a subfield of machine learning where the primary goal is to train reinforcement learning agents to perform tasks by incorporating feedback from humans. This feedback can come in various forms, such as binary rewards (e.g., right or wrong), scalar rewards, preferences between actions, demonstrations, natural language instructions, or corrections to the agent's behavior.

The idea behind reinforcement learning from human feedback is to leverage human intuition and decision-making abilities to guide the learning process of the agent. In traditional reinforcement learning, an agent learns to make decisions by interacting with an environment and receiving rewards or punishments based on the outcomes of those interactions. However, defining a suitable reward function can be challenging and time-consuming, often leading to reward shaping or reward hacking, where the agent finds unexpected and often unintended ways to maximize the reward.

By using human feedback, reinforcement learning can address some of these issues by allowing humans to directly influence the learning process. Human trainers can provide real-time feedback, scale their feedback to match the complexity of the task, and update their feedback based on the agent's progress. This approach can lead to more robust and generalized agents, as they are guided not by handcrafted reward functions but by the nuanced understanding and experience of human trainers.

However, reinforcement learning from human feedback faces several challenges, such as the high cost of obtaining human feedback, subjective and inconsistent feedback, scaling issues, and methods for efficient and effective integration of human feedback into learning algorithms. Despite these challenges, the research in this area has shown promising results in various applications, including robotics, video games, prosthetics control, natural language processing, and many more.
```

그리고 [Llama 3.1 Tulu 3 70B DPO](https://huggingface.co/allenai/Llama-3.1-Tulu-3-70B-DPO)의 RLHF 이후 완성:
```text
Reinforcement learning from human feedback (RLHF) is a subset of reinforcement learning (RL) where the rewards that guide the learning process are provided by humans, rather than a pre-specified numerical reward function. This is particularly useful in situations where it is hard or infeasible to design a suitable reward function, or where the desired performance involves subjective judgments that humans are uniquely equipped to provide.

In traditional reinforcement learning, an agent interacts with an environment. With each interaction, the agent performs an action, receives an observation, and gets a numerical reward that tells it how well it performed in that situation. Through many such interactions, the agent learns to maximize the cumulative reward.

In contrast, in reinforcement learning from human feedback:

1. **Human Input:** Instead of a predefined reward function, the feedback comes from human evaluators, who provide qualitative, subjective judgments about the agent's performance. This feedback can be in various forms, such as numerical ratings, binary preference signals, natural language feedback, or even demonstrations showing how to improve.

2. **Feedback Loop:** Human evaluators observe the agent's actions or outcomes and provide feedback in real-time (or close to real-time). The agent learns through this iterative feedback loop, gradually adapting its behavior to align more closely with human preferences.

3. **Reward Shaping:** Because raw human feedback is not always directly usable in learning algorithms due to variability and potential bias, a reward shaping process may be involved, whereby human feedback is converted into a more stable signal that guides learning.

4. **Preference-based RL:** One common approach within RLHF is preference-based reinforcement learning, where the agent is trained by comparing pairs of actions (or outcomes) and receiving human feedback on which one is better. From these pairwise comparisons, the agent infers a reward function that it can use to guide its actions.

5. **Inverse Reinforcement Learning (IRL):** Another technique used in RLHF is IRL, which aims to infer the underlying reward function that could explain human actions or decisions, and then use that reward function to train the agent.

**Applications:**

Reinforcement Learning from human feedback has found applications in:

- **Robotics:** Teaching robots to perform tasks that are difficult to specify with predefined reward functions.
- **Game playing:** Human feedback can guide AI agents to better emulate human-like strategies or behaviors.
- **Language and content generation:** For example, suggestively guiding AI models to generate more coherent, creative, or helpful content.
- **Recommendation systems:** Human feedback can help adjust and refine the recommendations provided to users.

**Challenges:**

Using human feedback introduces some specific challenges:

1. **Scalability:** Human feedback is time-consuming and expensive to collect, and it's hard to scale to complex or high-dimensional tasks.
2. **Bias and Noise:** Human feedback can be noisy, inconsistent, and biased, leading to suboptimal learning outcomes.
3. **Alignment:** Ensuring that the reinforcement learning agent aligns with the true preferences of the human evaluator(s) and not just the superficial feedback, is difficult.
4. **Safety and Misuse:** There can be concerns about the safety and ethical implications of using potentially flawed human feedback, especially if the feedback process is corrupted or gamed.

Researchers are actively working on techniques to mitigate these challenges and improve RLHF approaches. One such approach is leveraging machine learning models to amplify the efficacy of the human feedback, for example, by learning to generalize from sparse human input or by synthesizing plausible feedback in situations where direct human input is not available.
```

DPO 모델의 응답은 두 모델이 동일한 사전 학습된 기반 모델로 학습되었으므로 유사한 정보를 갖지만, 선호도 미세조정은 인간 선호도의 일반적인 개념과 반복적으로 연관되는 것으로 나타난 더 많은 구조, 형식, 그리고 당연히 길이를 추가한다.
이와 같은 더 많은 예시들은 [https://www.rlhfbook.com/library](https://rlhfbook.com/library)의 이 책과 관련된 완성 라이브러리에서 확인할 수 있다.

## 수다스러움의 균형

RLHF 또는 선호도 미세조정 방법들은 AlpacaEval 및 기타 자동 대화 리더보드 (에이전트가 단순한 대화 과제에서 얼마나 도움이 되고, 무해하며, 정직한지를 근사하기 위해 LLM-as-a-judge를 사용하는) 같은 점수를 높이기 위해 수많은 사람들이 사용하고 있지만, RLHF가 여기서 제공하는 대규모 이득은 Arena처럼 게임화하기 더 어려운 평가에서 점수를 비례적으로 이동시키지 않는다.
긴장감은 RLHF 방법들이 이러한 모델들에서 측정 가능한 향상을 제공하는 반면, 그 학습이 사람들이 실제로 신경 쓰는 성능으로 항상 전환되지는 않는다는 것이다.
RLHF 문헌의 확립을 통해, 많은 모델들이 RLHF로 모델의 "정렬 (alignment)"을 높이기 위한 관련 방법들과 함께 출시되었지만, 종종 너무 지나쳐서 오해를 불러일으키거나 무의미한 평가 점수를 발표했다.

정렬에 의해 동기화된 이러한 RLHF 방법들은 올바르게 수행될 때 모델을 작업하기 더 쉽고 즐겁게 만든다.
이것은 종종 MT Bench나 AlpacaEval 같은 평가 도구에서 명확한 향상을 동반한다.

2023년 가을, 직접 선호도 최적화 (DPO, direct preference optimization)와 선호도 미세조정을 위한 근위 정책 최적화 (PPO, proximal policy optimization) 및 기타 RL 기반 방법들과의 역할에 대한 논쟁이 절정에 달했으며 -- 대화 평가와 실제 세계 성능의 균형이 그 중심에 있었다 (기술적 트레이드오프에 대한 더 많은 논의는 8장, Ivison et al. 2024 [@ivison2024unpacking], 또는 [이 강연](https://youtu.be/YJMCSVLRUNs) 참조).
문제는 DPO와 PPO 같은 기법을 피드백 루프나 풍부한 데이터로 사용하여 수학이나 코딩 같은 다른 과제에서 모델을 실제로 심각하게 손상시켜 이 대화 성능을 위한 트레이드오프를 만들 수도 있다는 것이다.

DPO 대 PPO 논쟁이 확산되는 동안 놀라운 벤치마크를 발표하지만 모델 가중치가 없이 지속적인 공개 사용이 축적되지 않은 많은 논문들이 나왔는데, 이러한 모델들이 일반 사용에서 강건하지 않았기 때문이다.
2023년 가을 또는 그 직후에 RLHF를 적용할 때, 70억 파라미터 모델의 정렬된 버전이 종합적인 벤치마크에서 GPT-4를 실제로 이기는 방법은 없다 (이런 종류의 비교는 그 날의 작은 모델들이 최고의 대형 프런티어 모델을 강건하게 이길 수 없다는 점에서 유효하다).
명백해 보이지만, 이런 종류의 결과를 주장하는 논문들은 항상 있다.
@fig:DNO는 Direct Nash Optimization (DNO)이라는 논문에서 나온 것으로, 2024년 4월 7B 모델에서 AlpacaEval의 최첨단 상태임을 주장한다 [@rosset2024direct].
맥락으로, DNO는 정렬을 선호도 오라클에 대해 내쉬 균형 (Nash equilibrium)을 찾는 것으로 프레임화하여 쌍별 선호도 (승률 격차)를 직접 최적화하는 보상 모델+PPO (고전적 RLHF) 또는 일회성 DPO에 대한 배치 기반, 온-정책 *반복적* 대안이다.
학문적 인센티브가 더 넓은 사회에서 극도로 관심을 받는 기술과 상호작용할 때 이러한 도전들이 발생한다.

![GPT-4와 같은 것들을 능가하는 소형 모델을 강조하는 Direct Nash Optimization (DNO) 논문의 결과. Rosset et al. 2024. 라이선스 CC-BY.](images/dno-figure.png){#fig:DNO width=550px}

심지어 2024년 1월의 선구적인 논문 Self Rewarding Language Models [@yuan2025selfrewardinglanguagemodels]도 Llama 2 70B에서 비현실적으로 강한 점수를 공개했다.
당시에는 물론 70B 모델이 7B 모델보다 GPT-4에 더 가까울 수 있었다 (2024년의 인상적인 Llama 3 출시에서 보듯이), 하지만 최신 RLHF 논문의 주장에서 모델의 현실을 분리하는 것이 중요하다.
이러한 모델들은 좁은 테스트 세트에 맞춰 조정되어 있어 이기겠다고 주장하는 훨씬 더 큰 모델들과 실제 사용에서 잘 버티지 못한다.
더 많은 방법들이 이와 유사하게 나타났다 사라졌는데, 귀중한 통찰과 과장된 결과를 공유하여 RLHF를 이해하기 어렵게 만든다.

"이상한 RLHF"가 적용된 모델들의 증상은 종종 길이 편향이었다.
이것이 너무 일반적이 되어 AlpacaEval과 WildBench 같은 여러 평가 시스템 모두에 선형 길이 교정 메커니즘이 내장되었다.
이것은 'GPT-4를 이기기 위해' 또는 그 날의 선도 프런티어 모델을 이기기 위해 수다스러움에 과도하게 집중하는 인센티브를 패치하고, 짧고 유용한 모델이 실제로 이길 수 있는 덜 게임화된 역학을 만든다.

그럼에도 불구하고, 대화 모델을 수다스러움만을 위해 정렬하는 것은 이제 문헌에서 약간의 평판 부담을 갖고 있으며, 이러한 좁은 방법들이 다른 방식으로 모델을 손상시킬 수 있다는 것이 인정된다.
2023년 원래의 Alibaba Qwen 모델에서 나온 이 주석은 초기 정렬 실험에서 여러 번 관찰된 것으로, 수다스러움과 성능 간의 트레이드오프를 과장한다 [@qwen].

> 우리는 대량의 데이터로 모델을 사전 학습했고, 지도 미세조정 (SFT)과 직접 선호도 최적화 (DPO) 모두로 모델을 사후 학습했다. 그러나 DPO는 인간 선호도 평가에서는 향상을 가져오지만 벤치마크 평가에서는 저하를 가져온다.

이 트레이드오프가 올바르게 수행된 초기의 좋은 예는 2024년 3월의 Starling Beta 같은 모델이다 [@zhu2024starling].
이것은 다른 대화 모델인 OpenChat [@wang2023openchat] (실제로는 완전히 다른 조직에서 학습된)에서 미세조정된 모델이다.
그것의 학습은 전적으로 k-wise 보상 모델 학습과 PPO 최적화에 집중하며, Arena에서 10단계를 올린다.
모델의 평균 응답 길이가 증가하지만, 실제로 인간 평가자들에게 도움이 될 만큼 충분히 좋은 방식으로 이루어진다.
Olmo 3와 같은 후기 예시들은 실제로 상당한 대화 학습을 거쳤다는 것이 문서화되어 있지만, 저자들은 LLM-as-a-judge 기반 대화 벤치마크에서 가장 높은 잠재적 체크포인트 대신 더 높은 수학, 코딩, 추론 점수를 가진 최종 모델 체크포인트를 선호한다 [@teamolmo2025olmo3].

자연스러운 질문은: RLHF가 모델 응답을 왜 더 길게 만드는가 하는 것이다.
근본적으로, Arena와 같은 평가들은 모델의 평균적인 사용자들이 짧고 간결한 응답보다 더 길고 완전한 답변을 선호한다는 것을 보여주었다.
더 긴 답변은 빠르게 평가하는 사용자들에게 더 철저하거나, 유용하거나, 심지어 신뢰할 수 있는 것처럼 느껴질 수 있다.
이것은 *모든* 사용자의 선호도를 나타내는 것이 아니지만, 이러한 모델들은 많은 데이터 레이블러의 평균 선호도에 맞게 학습되므로 RLHF는 모델을 더 장황하게 만드는 경향이 있다.
