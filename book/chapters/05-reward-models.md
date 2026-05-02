<!--
  Copyright (c) 2025-2026 Nathan Lambert.
  Licensed under CC BY-NC-SA 4.0:
  https://creativecommons.org/licenses/by-nc-sa/4.0/
  Full license: https://github.com/natolambert/rlhf-book/blob/main/LICENSE-CHAPTERS
-->
---
prev-chapter: "지시 조정"
prev-url: "04-instruction-tuning"
page-title: 보상 모델
search-title: "5장: 보상 모델"
next-chapter: "강화학습"
next-url: "06-policy-gradients"
lectures:
  - video: "https://www.youtube.com/watch?v=4gIwiSPmQkU&list=PLL1tdVxB1CpVpEtMHxwuR4uI4Lxjw00_y&index=3"
    label: "강의 2: IFT, 보상 모델링, 거부 샘플링 (4, 5, 9장)"
---

# 보상 모델링

보상 모델 (reward model)은 복잡한 인간 선호도 (human preference)가 학습되는 곳이라는 점에서 현대적인 인간 피드백 기반 강화학습 (RLHF, Reinforcement Learning from Human Feedback) 접근법의 핵심이다.
보상 모델은 명시하기 어려운 신호로부터 우리의 모델이 학습할 수 있게 해준다.
보상 모델은 데이터의 복잡한 특징들을 다운스트림 훈련에 사용할 수 있는 표현으로 압축하는데—이는 현대 딥러닝의 복잡한 역량을 다시 한번 보여주는 일종의 마법이다.
이 모델들은 이후 장에서 연구하는 것처럼 핵심 최적화의 대리 목적함수 (proxy objective)로 작동한다.
@fig:rm-role-in-rlhf에 나타난 것처럼, 보상 모델은 표준 강화학습 (RL) 환경 역할과 유사하게 에이전트를 위한 학습 신호를 제공하지만, 고정된 환경과 달리 인간 선호도로부터 학습할 수 있다.

보상 모델은 역사적으로 환경 보상에 대한 대리 역할을 하는 강화학습 연구에서 광범위하게 사용되어 왔다 [@sutton2018reinforcement].
보상 모델은 현대적 형태로 가치 정렬 (value alignment) 문제를 연구하기 위한 도구로 제안되었다 [@leike2018scalable].
이 모델들은 일반적으로 어떤 종류의 입력을 받아 단일 스칼라 보상 (reward) 값을 출력하는 경향이 있다.
이 보상은 여러 형태를 취할 수 있다—전통적인 RL 문제에서는 문제에 대한 정확한 환경 보상을 근사하려 했지만, RLHF에서 보상 모델은 실제로 특정 입력이 "고품질"일 확률(즉, 쌍별 선호도 관계에서 선택된 답)을 출력한다.
RLHF를 위한 보상 모델링의 실천은 역강화학습 (inverse reinforcement learning)과 밀접하게 관련되어 있다. 역강화학습은 행동 궤적 (trajectory)이 주어졌을 때 에이전트의 보상 함수를 근사하는 문제이며 [@ng2000algorithms], 딥 강화학습의 다른 영역과도 연관된다.
높은 수준의 문제 진술은 동일하지만, 구현 및 초점 영역이 완전히 다르기 때문에 종종 완전히 분리된 연구 영역으로 간주된다.

가장 일반적인 보상 모델, 흔히 Bradley-Terry 보상 모델이라고 불리며 이 장의 주요 초점인 이 모델은 텍스트 한 조각이 훈련 비교에서 "선호되는" 텍스트에 가까울 확률을 예측한다.
이 섹션의 후반부에서 결과 보상 모델 (ORM, Outcome Reward Model), 과정 보상 모델 (PRM, Process Reward Model), 그리고 다른 유형의 보상 모델과 비교한다.

*이 장 전체에서 $x$는 프롬프트를, $y$는 완성 (completion)을 나타낸다. 이 표기법은 언어 모델 문헌에서 일반적으로 사용되며, 여기서 방법들은 개별 토큰이 아닌 전체 프롬프트-완성 쌍에 대해 동작한다.*

![RLHF에서 보상 모델은 표준 RL에서 보상을 반환하는 환경 구성 요소의 역할을 한다. 핵심 차이점은 RLHF에서는 환경에 의해 고정되는 것이 아니라 인간 선호도로부터 이 보상 함수를 제어하고 학습할 수 있다는 점이다.](images/rlhf-overview.png){#fig:rm-role-in-rlhf}

## Bradley-Terry 보상 모델 훈련

보상 모델의 표준적인 구현은 선호도의 Bradley-Terry 모델 [@BradleyTerry]에서 파생된다.
표준 RLHF 보상 모델을 훈련하는 방법에 대한 두 가지 인기 있는 표현이 있는데—이들은 수학적으로 동등하다.
시작하기 위해, Bradley-Terry 선호도 모델은 두 항목 $i$와 $j$ 사이의 쌍별 비교에서 심판이 $j$보다 $i$를 선호할 확률을 다음과 같이 정의한다:

$$P(i > j) = \frac{p_i}{p_i + p_j}.$$ {#eq:bradterry}

Bradley-Terry 모델은 각 항목이 잠재적 강도 $p_i > 0$을 가지며, 관측된 선호도는 이러한 기저 강도의 잡음 있는 반영이라고 가정한다.
$p_i = e^{r_i}$인 무한 점수로 Bradley-Terry 모델을 재매개변수화하는 것이 일반적이며, 다음과 같은 형태가 된다:

$$P(i > j) = \frac{e^{r_i}}{e^{r_i} + e^{r_j}} = \sigma(r_i-r_j).$$ {#eq:bradterry_unbounded}

점수의 차이만이 중요하다: 모든 $r_k$에 동일한 상수 $c$를 더해도 $P(i > j)$는 변하지 않는다.
이러한 형태들은 자연의 법칙이 아니라, RLHF에서 종종 잘 작동하는 인간 선호도의 유용한 근사치이다.

보상 모델을 훈련하려면 위의 관계를 만족하는 손실 함수 (loss function)를 공식화해야 한다.
실제로 이는 언어 모델을 스칼라 점수를 출력하는 모델로 변환함으로써 수행되며, 종종 모델의 최종 은닉 상태 (hidden state)로부터 단일 보상 값을 생성하는 작은 선형 헤드 (linear head)를 통해 이루어진다.
프롬프트 $x$와 두 개의 샘플링된 완성 $y_1$, $y_2$가 주어지면, 보상 모델 $r_\theta$로 두 개를 모두 점수화하고 조건부 점수를 $r_\theta(y_i \mid x)$로 표기한다.

보상 모델이 $y_1$이 $y_2$보다 선호될 확률을 다음과 같이 나타낸다:

$$P(y_1 > y_2 \mid x) = \frac{\exp\left(r_\theta(y_1 \mid x)\right)}{\exp\left(r_\theta(y_1 \mid x)\right) + \exp\left(r_\theta(y_2 \mid x)\right)}.$$ {#eq:bradterryrm}

선호된 완성을 $y_c$ (선택된 완성, chosen)로, 거부된 완성 (rejected completion)을 $y_r$로 나타낸다.

결과 손실은 시그모이드 (sigmoid)를 사용하여 점수 차이를 확률로 변환하면서 보상 모델이 거부된 것보다 인간이 선호한 완성에 더 높은 점수를 부여하도록 장려한다.
@eq:bradterryrm의 선호도 우도 (preference likelihood)가 출발점이다. 먼저 그 우도를 시그모이드 형태로 재작성하고, 마지막 단계에서만 보상 모델 훈련에 사용되는 동등한 음의 로그 우도 (NLL, negative log-likelihood) 손실로 변환한다:

$$
\begin{aligned}
\theta^* = \arg\max_\theta P(y_c > y_r \mid x) &= \arg\max_\theta \frac{\exp\left(r_\theta(y_c \mid x)\right)}{\exp\left(r_\theta(y_c \mid x)\right) + \exp\left(r_\theta(y_r \mid x)\right)} \\
&= \arg\max_\theta \frac{\exp\left(r_\theta(y_c \mid x)\right)}{\exp\left(r_\theta(y_c \mid x)\right)\left(1 + \frac{\exp\left(r_\theta(y_r \mid x)\right)}{\exp\left(r_\theta(y_c \mid x)\right)}\right)} \\
&= \arg\max_\theta \frac{1}{1 + \frac{\exp\left(r_\theta(y_r \mid x)\right)}{\exp\left(r_\theta(y_c \mid x)\right)}} \\ 
&= \arg\max_\theta \frac{1}{1 + \exp\left(-(r_\theta(y_c \mid x) - r_\theta(y_r \mid x))\right)} \\
&= \arg\max_\theta \sigma \left( r_\theta(y_c \mid x) - r_\theta(y_r \mid x) \right) \\
&= \arg\min_\theta - \log \left( \sigma \left(r_\theta(y_c \mid x) - r_\theta(y_r \mid x)\right) \right)
\end{aligned}
$$ {#eq:bradterryrm_deriv}

첫 번째 형태는 [@ouyang2022training] 및 다른 연구들에서와 같이 위에서 유도된 로그-시그모이드 표현이다:
$$\mathcal{L}(\theta) = - \log \left( \sigma \left( r_{\theta}(y_c \mid x) - r_{\theta}(y_r \mid x) \right) \right)$$ {#eq:rewardmodeling1}

두 번째는 [@askell2021general] 및 다른 연구들에서와 같이 소프트플러스 함수 $\log(1+e^x)$를 사용하여 표현된 수학적으로 동등한 형태이다:
$$\mathcal{L}(\theta) = \log \left( 1 + e^{r_{\theta}(y_r \mid x) - r_{\theta}(y_c \mid x)} \right)$$ {#eq:rewardmodeling2}

이들은 $\Delta = r_{\theta}(y_c \mid x) - r_{\theta}(y_r \mid x)$로 놓고 $\sigma(\Delta) = \frac{1}{1 + e^{-\Delta}}$를 사용하면 동등하며, 이는 $-\log\sigma(\Delta) = \log(1 + e^{-\Delta}) = \log\left(1 + e^{r_{\theta}(y_r \mid x) - r_{\theta}(y_c \mid x)}\right)$를 의미한다.
두 형태 모두 RLHF 문헌에 등장한다.

![선호도 보상 모델 훈련에는 선택된 완성과 거부된 완성의 쌍이 필요하다. 모델은 시퀀스 수준 표현, 종종 시퀀스 종료(EOS) 토큰의 은닉 상태로부터 각 완성에 대한 스칼라 점수를 계산하며, 대조 손실은 두 점수의 차이에만 의존한다.](images/pref_rm_training.png){#fig:pref_rm_training}

## 기본 보상 모델 아키텍처

보상 모델이 구현되는 가장 일반적인 방법은 Transformers의 `AutoModelForSequenceClassification`과 유사한 추상화를 통해서이다. 이는 언어 모델에 작은 선형 헤드를 추가하여 훈련 또는 추론 시 프롬프트-완성 쌍에 대한 스칼라 보상 점수를 생성한다.
추론 시, 모델은 *텍스트 조각이 선택될 상대적 확률*을 모델의 단일 로짓 (logit)으로 출력한다.

최종 임베딩 (embedding)에서 직접 선형 레이어를 취하는 것과 같은 다른 구현 옵션도 있지만, 오픈 툴링에서는 덜 일반적이다.

## 구현 예시

보상 모델링 손실을 구현하는 것은 매우 간단하다.
구현의 더 큰 도전은 별도의 데이터 로더와 추론 파이프라인을 설정하는 것이다.
완성이 있는 토큰화된 선택 및 거부 프롬프트가 있는 올바른 데이터 로더가 주어지면, 손실은 다음과 같이 구현된다:
```python
import torch.nn as nn
# inputs_chosen / inputs_rejected include the prompt tokens x and the respective
# completion tokens (y_c or y_r) that the reward model scores jointly.
rewards_chosen = model(**inputs_chosen)
rewards_rejected = model(**inputs_rejected)

loss = -nn.functional.logsigmoid(rewards_chosen - rewards_rejected).mean()
```

더 큰 그림에서, 이는 종종 인과 언어 모델 (causal language model, 각 토큰을 이전의 모든 토큰에 조건화하여 좌에서 우로 예측하는 모델) 내에 있으며, 입력의 점수로 최종 은닉 상태에서 전환하는 추가 헤드가 (위의 손실로 학습되어) 추가된다.
코드는 표준 트랜스포머 (transformer) 입력—`input_ids` (토큰화된 텍스트)와 `attention_mask` (실제 토큰 대 패딩을 표시)—을 받아 마지막 실제 토큰에서 은닉 상태(입력의 모델 내부 표현)를 추출하고, 이를 선형 레이어에 통과시켜 스칼라 보상을 생성한다.
이 모델은 다음과 같은 구조를 가진다:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class BradleyTerryRewardModel(nn.Module):
    """
    Standard scalar reward model for Bradley-Terry preference learning.

    Usage (pairwise BT loss):
        rewards_chosen = model(**inputs_chosen)    # (batch,)
        rewards_rejected = model(**inputs_rejected)  # (batch,)
        loss = -F.logsigmoid(rewards_chosen - rewards_rejected).mean()
    """
    def __init__(self, base_lm):
        super().__init__()
        self.lm = base_lm  # e.g., AutoModelForCausalLM
        self.head = nn.Linear(self.lm.config.hidden_size, 1)

    def _sequence_rep(self, hidden, attention_mask):
        """
        Get a single vector per sequence to score.
        Default: last non-padding token (EOS token); if no mask, last token.
        hidden: (batch, seq_len, hidden_size)
        attention_mask: (batch, seq_len)
        """

        # Index of last non-pad token in each sequence
        # attention_mask is 1 for real tokens, 0 for padding
        lengths = attention_mask.sum(dim=1) - 1  # (batch,)
        batch_idx = torch.arange(hidden.size(0), device=hidden.device)
        return hidden[batch_idx, lengths]  # (batch, hidden_size)

    def forward(self, input_ids, attention_mask):
        """
        A forward pass designed to show inference structure of a standard reward model.
        To train one, this function will need to be modified to compute rewards from both
         chosen and rejected inputs, applying the loss above.
        """
        outputs = self.lm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        # Final hidden states: (batch, seq_len, hidden_size)
        hidden = outputs.hidden_states[-1]

        # One scalar reward per sequence: (batch,)
        seq_repr = self._sequence_rep(hidden, attention_mask)
        rewards = self.head(seq_repr).squeeze(-1)

        return rewards
```

이 섹션과 이어지는 내용에서, 보상 모델(및 후처리 학습의 대부분)의 구현 복잡성은 데이터 로더를 올바르게 구성하고 분산 학습 시스템을 구축하는 것에 있다.
보상 모델을 훈련할 때 가장 일반적인 관행은 과적합 (overfitting)을 피하기 위해 에폭 (epoch) 1회만 훈련하는 것임을 주의하라.

## 보상 모델 변형

보상 모델링은 RLHF에서 상대적으로 덜 탐구된 영역이다.
전통적인 보상 모델링 손실은 많은 인기 있는 연구에서 수정되었지만, 그 수정들이 단일 모범 사례로 수렴되지는 않았다.

### 선호도 마진 손실

주석자 (annotator)가 리커트 척도 (Likert Scale, 선호도 크기를 나타내는 순서가 있는 범주를 가진 평가 척도, 예: 1--5)로 점수나 순위를 제공하는 경우, 관계적 수량의 크기를 훈련에 사용할 수 있다.
가장 일반적인 관행은 상대적 평가나 순위 강도의 혼합 정보를 선택된 완성과 거부된 완성으로만 줄여 선호도 방향으로 데이터를 이진화하는 것이다.
선호도의 크기와 같은 추가 정보는 모델 훈련을 개선하는 데 사용되었지만, 표준 관행으로 수렴되지는 않았다.
Llama 2는 선호도의 크기를 구분하기 위해 두 데이터 포인트 사이의 마진 $m(y_c, y_r)$를 사용할 것을 제안한다:

$$\mathcal{L}(\theta) = - \log \left( \sigma \left( r_{\theta}(y_c \mid x) - r_{\theta}(y_r \mid x) - m(y_c, y_r) \right) \right)$$ {#eq:rewardmodelingmargin}

예를 들어, 각 완성에는 종종 품질 측면에서 1에서 5까지의 순위가 부여된다.
선택된 샘플에 5점, 거부된 샘플에 2점이 부여된 경우, 마진은 $m(y_c, y_r)= 5 - 2 = 3$이 된다.
마진을 계산하기 위한 다른 함수도 탐색할 수 있다.

Llama 3에서는 팀이 스케일링 후 개선 효과가 감소하는 것을 관찰하면서 마진 항이 제거되었음을 주목하라.

### 프롬프트당 다중 비교 균형 맞추기

InstructGPT는 순위를 매기기 위해 프롬프트당 $K = 4$에서 9개의 완성을 사용하는 영향을 연구하며, 각 프롬프트에서 $\binom{K}{2}$개의 쌍별 비교를 생성한다 [@ouyang2022training].
이러한 비교들은 동일한 프롬프트를 공유하기 때문에 높은 상관관계를 가지므로, 이를 데이터셋에 단순히 섞으면 보상 모델이 과적합된다.
이를 해결하기 위해 비교당 프롬프트당 손실 업데이트에 가중치를 부여한다—재가중치 없이는 완성이 더 많은 프롬프트가 더 많은 쌍을 생성하기 때문에 단순히 더 많은 총 손실에 기여할 것이다.
실제로 단일 프롬프트의 모든 $\binom{K}{2}$ 비교는 일반적으로 동일한 훈련 배치에 포함되고 함께 평균화되므로, 각 프롬프트는 여러 별도 배치에 걸쳐 나타나는 대신 하나의 그룹화된 업데이트에 기여한다.
이는 개별 프롬프트에 대한 과적합을 줄이고 더 많은 완성이 샘플링된 프롬프트가 손실을 지배하는 것을 방지한다.
손실 함수는 다음과 같다:

$$\mathcal{L}(\theta) = - \frac{1}{\binom{K}{2}} \mathbb{E}_{(x, y_c, y_r)\sim D} \log \left( \sigma \left( r_{\theta}(y_c \mid x) - r_{\theta}(y_r \mid x) \right) \right)$$ {#eq:rewardmodelinginstructgpt}


### K-방식 손실 함수

RLHF에 적합한 인간 선호도 모델을 만들 수 있는 다른 많은 공식화가 있다.
그러한 예 중 하나로, 인기 있는 초기 RLHF 모델인 Starling 7B와 34B [@zhu2024starling]에서 사용된 Plackett-Luce 모델 [@liu2019learning]을 기반으로 한 K-방식 손실 함수가 있다.

Zhu et al. 2023 [@zhu2023principled]은 다음과 같이 설정을 공식화한다.
프롬프트 또는 상태 $s^i$와 함께, $K$개의 행동 $(a_0^i, a_1^i, \cdots, a_{K-1}^i)$이 $P(a_0,\cdots,a_{K-1}|s^i)$에서 샘플링된다.
그런 다음, 레이블러를 사용하여 $\sigma^i: [K] \mapsto [K]$로 선호도를 순위화하며, 여기서 $\sigma^i$는 행동 순위를 나타내는 함수이고 $\sigma^i(0)$이 가장 선호되는 행동이다. 이는 모든 $K$개 항목의 완전한 순위에 대한 Plackett-Luce 확률을 산출한다:

$$P(\sigma^i|s^i,a_0^i,a_1^i,\ldots,a_{K-1}^i) = \prod_{k=0}^{K-1} \frac{\exp(r_{\theta\star}(s^i,a_{\sigma^i(k)}^i))}{\sum_{j=k}^{K-1}\exp(r_{\theta\star}(s^i,a_{\sigma^i(j)}^i))}$$ {#eq:kwise_rm}

$K = 2$일 때, 이는 쌍별 비교에 대한 Bradley-Terry (BT) 모델로 귀결된다.
어쨌든, 일단 훈련되면 이러한 모델들은 RLHF 훈련 중에 다른 보상 모델과 유사하게 사용된다.


## 결과 보상 모델

<!-- Huge thanks to Hangliang Ren, graduate student at Northeastern University for helping with this section (and PRMs), see https://github.com/myhott163com/RLHF_ORM_PRM -->

언어 모델 및 다른 AI 시스템에 대한 *선호도 조정 (preference tuning)*의 대부분은 위에서 논의한 Bradley-Terry 모델로 이루어진다.
추론이 많이 필요한 작업에서는 결과 보상 모델 (ORM, Outcome Reward Model)을 사용할 수 있다.
ORM의 훈련 데이터는 표준 선호도 조정과 유사한 방식으로 구성된다.
여기서는 문제 진술 또는 프롬프트 $x$와 두 개의 완성 $y_1$, $y_2$가 있다.
여기서 사용되는 귀납적 편향 (inductive bias)은 하나의 완성이 문제에 대한 올바른 해결책이고 다른 하나는 잘못된 것이어야 한다는 것으로, $(y_c, y_{ic})$를 생성한다.

사용되는 모델의 아키텍처는 단일 로짓을 출력할 수 있는 모델에 선형 레이어가 추가된 표준 보상 모델과 매우 유사하다 (RM의 경우)—ORM의 경우, 이어지는 훈련 목적함수가 약간 다르다 [@cobbe2021gsm8k]:

> [우리는] 모델이 모델 완성을 올바르거나 잘못된 것으로 레이블하는 것을 학습하는, 원래의 언어 모델링 목적함수에 추가한 공동 목적함수로 검증자(verifier)를 훈련시킨다.
> 아키텍처적으로, 이는 우리의 검증자가 언어 모델임을 의미하며, 토큰별로 예측을 출력하는 작은 스칼라 헤드를 가진다.
> 우리는 이 스칼라 헤드를 언어 모델의 최종 역임베딩 (unembedding) 레이어가 출력하는 로짓에 작동하는 단일 편향 파라미터와 단일 이득 파라미터로 구현한다.

이를 번역하면, 전체 시퀀스에 대해 하나의 로짓을 출력하는 전통적인 RM의 분류 헤드 대신 토큰별로 두 클래스(1은 올바름, 0은 잘못됨)를 예측할 수 있는 언어 모델링 헤드로 구현된다.
형식적으로, [@lyu2025exploring]에 따르면 이는 토큰별 이진 교차 엔트로피 (binary cross-entropy) 손실이다:

$$\mathcal{L}_{\text{CE}}(\theta) = -\mathbb{E}_{(s,r)\sim \mathcal{D}}[r\log p_\theta(s) + (1-r)\log(1-p_\theta(s))]$$ {#eq:orm_loss}

여기서 $r \in \{0,1\}$은 이진 레이블로 1은 주어진 프롬프트에 대한 올바른 답에 적용되고 0은 잘못된 답에 적용되며, $p_\theta(s)$는 훈련 중인 모델로부터의 정확도의 예측 확률에 비례하는 스칼라이다.
코드에서 이 결과 레이블은 모든 완성 토큰에 복사되고, 프롬프트 토큰은 `-100`으로 마스킹되어 손실에 기여하지 않는다.

결과 보상 모델(그리고 과정 보상 모델에서 볼 것처럼 다른 유형들)을 구현하는 것은 완성이 올바른 샘플인지에 따라 토큰별로 교차 엔트로피 손실을 적용하는 것을 포함한다.
이는 표준 Bradley-Terry 보상 모델의 대조적인 시퀀스 수준 손실이 필요한 선택-거부 구조가 필요 없는 언어 모델링 손실에 훨씬 더 가깝다.
아래의 단순화된 ORM 훈련 설정에서, 우리는 새로운 토큰을 샘플링하거나 다음 토큰 예측에 대해 LLM을 훈련하지 않는다; 고정된 프롬프트-완성 시퀀스를 백본에 공급하고 ORM 헤드가 정확도 레이블을 예측하도록 훈련한다.

모델 구조는 다음과 같을 수 있다:

```python
import torch.nn as nn
import torch.nn.functional as F

class OutcomeRewardModel(nn.Module):
    def __init__(self, base_lm):
        super().__init__()
        self.lm = base_lm  # e.g., AutoModelForCausalLM
        self.head = nn.Linear(self.lm.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask=None, labels=None):
        """
        input_ids contains a full prompt+completion sequence.
        labels is token-aligned: prompt tokens are -100, and each completion
         token repeats the sequence outcome label (1=correct, 0=incorrect).
        If labels=None, this is an inference-only forward pass and the loss is
         returned as None.
        """
        outputs = self.lm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        # Final hidden states: (batch, seq_len, hidden_size)
        hidden = outputs.hidden_states[-1]
        # One scalar logit per token: (batch, seq_len)
        logits = self.head(hidden).squeeze(-1)

        # Inference-only forward pass: no loss is computed.
        if labels is None:
            return None, logits
        # Only compute loss on completion tokens (labels 0 or 1)
        # Prompt tokens have labels = -100
        mask = labels != -100
        if mask.any():
            loss = F.binary_cross_entropy_with_logits(
                logits[mask], labels[mask].float()
            )
        return loss, logits
```

단순화된 손실 함수는 다음과 같다:

```python
# Feed the full prompt+completion sequence once; no token sampling happens here.
# Assume model already has: model.lm (backbone) + model.head
hidden = model.lm(**inputs, output_hidden_states=True).hidden_states[-1]
logits_per_token = model.head(hidden).squeeze(-1)  # (batch, seq_len)
# This will sometimes be compressed as model.forward() in other implementations

# Binary labels: 1=correct, 0=incorrect (prompt tokens masked as -100)
mask = labels != -100
loss = F.binary_cross_entropy_with_logits(
    logits_per_token[mask], labels[mask].float()
)
```

여기서 중요한 직관은 ORM이 시퀀스의 모든 토큰에서 정확도 확률을 출력한다는 것이다 (최종 답에 의해서만 판단됨—추론 오류는 ORM 훈련 과정에서 포착되지 않는다).
업데이트와 손실이 결과 및 어텐션 맵핑에 따라 토큰별로 전파되므로 이는 잡음이 많은 과정일 수 있다.

![추론 시, 결과 보상 모델은 토큰별 정확도 확률을 출력한다. 프롬프트 토큰은 마스킹되고 (예: label=-100), 각 완성 토큰은 모델이 응답이 올바른 답으로 이어진다고 믿는지 여부를 나타내는 확률을 받는다.](images/orm_inference.png){#fig:orm_inference}

![결과 보상 모델 훈련은 검증자 또는 데이터셋의 오프라인 레이블을 사용한다 (예: 올바른 완성에는 모두 1). 각 완성 토큰은 결과 레이블에 대해 이진 교차 엔트로피로 훈련되며, 토큰별 확률은 검증, 필터링, 또는 재순위화를 위한 최종 점수로 집계된다.](images/orm_training.png){#fig:orm_training}

이러한 모델들은 계속 사용되어 왔지만 오픈 소스 RLHF 도구에서는 지원이 적다.
예를 들어, 동일한 유형의 ORM이 *Let's Verify Step by Step* [@lightman2023let]의 획기적인 연구에서 사용되었지만, 손실의 언어 모델링 예측 부분 없이 사용되었다.
그런 다음 최종 손실은 최종 답이 올바른지 예측하는 모든 토큰에 대한 교차 엔트로피 손실이다.

지원 부족으로 인해, 결과 보상 모델 (ORM)이라는 용어는 여러 방식으로 사용되어 왔다.
일부 문헌, 예를 들어 [@lyu2025exploring]은 Cobbe et al. 2021의 원래 정의를 계속 사용하는 반면, 다른 문헌은 완성이 올바른지 예측하도록 훈련된 모든 검증자에 대해 더 광범위하게 사용한다.


## 과정 보상 모델

과정 보상 모델 (PRM, Process Reward Model)은 원래 과정 지도 보상 모델 (process-supervised reward model)이라 불렸으며, 사고의 연쇄 (CoT, chain-of-thought) 추론 과정의 모든 *단계*에서 점수를 출력하도록 훈련된 보상 모델이다.
이는 EOS 토큰에서만 점수를 출력하는 표준 RM이나 모든 토큰에서 점수를 출력하는 ORM과 다르다.
과정 보상 모델은 각 추론 단계의 끝에서 감독이 필요하며, 단계의 토큰이 관련 타겟으로 훈련되는 유사한 방식으로 훈련된다—타겟은 PRM의 경우 단계이고 ORM의 경우 전체 응답이다.

[@lightman2023let]에 따르면, 이진 레이블 PRM은 일반적으로 단계별 교차 엔트로피 손실로 최적화된다:

$$\mathcal{L}_{\text{PRM}}(\theta) = - \mathbb{E}_{(x, s) \sim \mathcal{D}} \left[ \sum_{i=1}^{K} y_{s_i} \log r_\theta(s_i \mid x, s_{< i}) + (1 - y_{s_i}) \log \left(1 - r_\theta(s_i \mid x, s_{< i})\right) \right] $$ {#eq:prm_loss}

여기서 $s$는 $K$개의 주석이 달린 단계를 가진 샘플링된 사고의 연쇄이고, $y_{s_i} \in \{0,1\}$은 $i$번째 단계가 올바른지를 나타내며, $r_\theta(s_i \mid x, s_{< i})$는 원래 프롬프트 $x$와 모든 이전 단계 $s_{< i}$에 조건화된 단계 $s_i$가 유효하다는 PRM의 예측 확률이다.

다음은 HuggingFace의 TRL (Transformer Reinforcement Learning) [@vonwerra2022trl]에서 가져온 트레이너에서 이 단계별 레이블이 어떻게 패키징될 수 있는지의 예시이다:

```python
# Get the ID of the separator token and add it to the completions
separator_ids = tokenizer.encode(step_separator, add_special_tokens=False)
completions_ids = [completion + separator_ids for completion in completions_ids]

# Create the label 
labels = [[-100] * (len(completion) - 1) + [label] for completion, label in zip(completions_ids, labels)]
```

전통적으로 PRM은 추론 단계의 끝, 예를 들어 이중 줄바꿈이나 다른 특수 토큰에 해당하는 토큰에서만 토큰을 출력하는 언어 모델링 헤드로 훈련된다.
이러한 예측은 일반적으로 잘못된 경우 -1, 중립인 경우 0, 올바른 경우 1이다.
이러한 레이블은 모델이 올바른 경로에 있는지 여부가 아니라, 단계가 올바른지와 반드시 연결되지는 않는다.

![과정 보상 모델은 단계 경계 (예: 줄바꿈 토큰)에서만 감독을 제공한다. 각 단계는 3-클래스 레이블을 받는다: 올바름 (+1), 중립 (0), 또는 잘못됨 (-1). 다른 모든 토큰은 훈련 중에 마스킹된다.](images/prm_training_inference.png){#fig:prm_training_inference}

PRM의 예시 구성은 아래에 나와 있다.

```python
import torch.nn as nn
import torch.nn.functional as F

class ProcessRewardModel(nn.Module):
    def __init__(self, base_lm, num_classes=3):
        super().__init__()
        self.lm = base_lm  # e.g., AutoModelForCausalLM
        self.head = nn.Linear(self.lm.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask=None, labels=None):
        """
        The inputs are tokenized prompts and completions, where the end of a
         "reasoning step" is denoted by a designated separator token such as a
         newline or other special marker rather than batch padding.
        labels will be a list of labels, True, False, and Neutral (3 labels) which
         will be predicted by the model.
        If labels=None, this is an inference-only forward pass and the loss is
         returned as None.
        """
        outputs = self.lm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        # Final hidden states: (batch, seq_len, hidden_size)
        hidden = outputs.hidden_states[-1]
        # One logit vector per token: (batch, seq_len, num_classes)
        logits = self.head(hidden)

        # Inference-only forward pass: no loss is computed.
        if labels is None:
            return None, logits
        # Only compute loss at step boundaries (where labels != -100)
        # Labels map: -1 -> 0, 0 -> 1, 1 -> 2 (class indices)
        mask = labels != -100
        if mask.any():
            loss = F.cross_entropy(
                logits[mask], labels[mask]
            )
        return loss, logits
```

핵심 손실 함수는 레이블이 다른 간격으로 적용되는 결과 보상 모델과 매우 유사하게 보인다.
```python
# Assume model outputs 3-class logits per token
hidden = model.lm(**inputs, output_hidden_states=True).hidden_states[-1]
logits = model.head(hidden)  # (batch, seq_len, 3)

# 3-class labels at step boundaries only: 0=-1, 1=0, 2=1 (others masked as -100)
mask = labels != -100
loss = F.cross_entropy(logits[mask], labels[mask])
```

## 보상 모델 유형 비교 (및 가치 함수)

다루어진 다양한 유형의 보상 모델들은 RLHF 및 다른 후처리 학습 방법에서 "품질"이 측정될 수 있는 다양한 방법의 스펙트럼을 나타낸다.
아래에 모델이 예측하는 것과 훈련 방법에 대한 요약이 있다.

::: {.table-wrap}
| 모델 클래스 | 예측하는 것 | 훈련 방법 | LM 구조 |
|------------|------------------|---------------------|--------------|
| **보상 모델 (Reward Models)** | 시퀀스 수준 품질 점수 $r_\theta(x, y)$ | 완성 사이의 쌍별 (또는 N-방식) 비교 간의 대조 손실 | EOS/마지막 토큰 은닉 상태에 대한 선형 헤드 |
| **결과 보상 모델 (Outcome Reward Models)** | 토큰별 답이 올바를 확률 | 레이블된 결과 쌍 (예: 검증 가능한 도메인에서의 성공/실패) | 토큰별 이진 교차 엔트로피 헤드; 레이블은 결과 레이블을 반복 |
| **과정 보상 모델 (Process Reward Models)** | 추론 단계 끝에서 중간 단계에 대한 보상 또는 점수 | 중간 피드백 또는 단계별 주석을 사용하여 훈련 (추론 단계의 토큰별 훈련) | 단계 정확도를 예측하는 토큰별 헤드 (-1, 0, 1) |
| **가치 함수 (Value Functions)** | 현재 상태가 주어졌을 때 예상 반환 | 시퀀스의 각 지점으로의 회귀를 통해 훈련 | 토큰별 출력이 있는 스칼라 회귀 헤드 |
표: 보상 모델 유형 비교. {#tbl:rm_compare}
:::

이 표의 구분에 대한 몇 가지 주의 사항으로, 모델 유형 간의 경계가 항상 명확하지는 않기 때문이다:

- 선호도 조정과 추론 훈련 모두에서, 가치 함수 (value function)는 종종 할인 계수 (discount factor) 1을 가지며, 이는 가치 함수를 결과 보상 모델에 더 가깝게 만들지만 훈련 손실이 다르다.
- 과정 보상 모델은 중간 상태에서 롤아웃 (rollout)을 수행하고 결과 데이터를 수집하여 감독될 수 있다. 이는 여러 아이디어를 혼합하지만, *손실*이 추론 단계 레이블별이라면 PRM으로 가장 잘 지칭된다.

**올바른/잘못된 쌍으로 Bradley-Terry 쌍별 모델을 훈련하면 어떻게 되는가?**
결과 보상 모델에 대한 혼란의 대부분은 답의 정확도에서 파생된 쌍별 데이터로 보상 모델을 훈련하는 소수의 문헌에서 비롯되었다.
이 도메인에서는 선택된 응답을 문제에 대한 올바른 답으로, 거부된 응답을 *동일한 문제*에 대한 잘못된 답으로 설정한다.
이는 기술적으로 ORM이 아니며 여전히 대조적인 시퀀스 수준 손실로 직접 훈련된다.
이는 기술적으로 여전히 Bradley-Terry 모델이며 우리가 다룬 첫 번째 모델 클래스에 해당한다.

**ORM 대 가치 함수 (Value Function).**
ORM과 가치 함수는 동일한 헤드 아키텍처로 토큰별 출력을 생성하기 때문에 유사하게 보일 수 있지만, *예측하는 것*과 *타겟이 어디서 오는지*에서 다르다:

- **ORM**은 즉각적인 토큰 로컬 수량을 예측한다: $p(\text{correct}_t)$ 또는 $r_t$. 타겟은 *오프라인 레이블* (토큰/시퀀스를 올바르거나 잘못된 것으로 표시하는 검증자 또는 데이터셋)에서 온다.
- **가치 함수**는 *남은* 예상 반환을 예측한다: $V(s_t) = \mathbb{E}[\sum_{k \geq t} \gamma^{k-t} r_k \mid s_t]$. 타겟은 일반적으로 현재 정책 $\pi_\theta$ 하에서 *온-정책 (on-policy) 롤아웃으로부터 계산되고*, 정책이 변경됨에 따라 변한다 (기술적으로, 가치 함수는 오프-정책 (off-policy)일 수도 있지만, 이는 언어 모델링 연구에서는 확립되지 않았다).

밀도 있는 토큰 보상 $r_t = \mathbb{1}[\text{token is correct}]$를 정의하고 $\gamma = 1$을 사용한다면, ORM은 $r_t$ (또는 $p(r_t = 1)$)를 학습하는 반면 가치 헤드는 남은 합산 $\sum_{k \geq t} r_k$를 학습한다.
이들은 동일한 기반 모델과 헤드 차원을 공유할 수 있지만, *의미론과 감독 파이프라인*이 다르다: ORM은 고정된 레이블로 오프라인에서 훈련되는 반면, 가치 함수는 온-정책으로 훈련되고 정책 그래디언트 (policy gradient)에 대한 이점 $A_t = \hat{R}_t - V_t$를 계산하는 데 사용된다.

### 보상 모델 유형별 추론

모델들은 추론 시 (훈련된 후) 데이터를 다르게 처리하며, RM이 사용되는 일련의 작업을 처리하기 위함이다.

**Bradley-Terry RM (선호도 모델):**

- *입력:* 프롬프트 $x$ + 후보 완성 $y$
- *출력:* EOS/마지막 토큰 은닉 상태로부터 선형 레이어를 통한 단일 스칼라 $r_\theta(x, y)$
- *사용:* $k$개 완성 재순위화, 상위 1개 선택 (최적-N 샘플링, best-of-N sampling); 또는 RLHF를 위한 터미널 보상 제공
- *집계:* 스칼라 출력으로 불필요

**결과 RM:**

- *입력:* 프롬프트 $x$ + 완성 $y$
- *출력:* 완성 토큰에 대한 토큰별 확률 $p_t \approx P(\text{correct at token } t)$
- *사용:* 완료된 후보 점수화; 평균, 최솟값 (꼬리 위험), 또는 곱 $\prod_t p_t$ (동등하게, 로그 확률 합산 $\sum_t \log p_t$)을 통해 집계
- *집계 선택:* 평균 정확도, 최솟값 $p_t$, 마지막 $m$ 토큰에 대한 평균, 또는 $p_t < \tau$이면 임계값 플래깅

**과정 RM:**

- *입력:* 프롬프트 $x$ + 단계 경계가 있는 추론 추적
- *출력:* 단계 경계에서의 점수 (예: 올바름/중립/잘못됨에 대한 클래스 로짓)
- *사용:* 완료된 사고의 연쇄 점수화; 또는 낮은 점수의 분기를 가지치기하여 탐색/디코딩 안내
- *집계:* 토큰이 아닌 단계에 대해—평균 단계 점수, 최솟값 (빠른 실패), 또는 이후 단계를 선호하는 가중 합계

**가치 함수:**

- *입력:* 프롬프트 $x$ + 현재 접두사 $y_{\leq t}$ (상태)
- 출력: 완성의 각 토큰 위치에서 $V_t$ (상태 $t$에서의 예상 남은 반환)
- 사용: RL 훈련 중 토큰별 이점 $A_t = \hat{R}_t - V_t$ 계산; 각 단계의 값은 기준선으로 사용
- *집계:* 일반적으로 마지막 생성된 토큰에서 $V$를 취함; 해석은 "정확도 확률"과 다름

요약하면, 다양한 모델을 이해하는 방법은 다음과 같다:

- **RM:** "이 전체 답변이 얼마나 좋은가?" → 스칼라 값
- **ORM:** "어떤 부분이 올바르게 보이는가?" → 토큰별 정확도
- **PRM:** "추론 단계가 타당한가?" → 단계별 점수
- **가치:** "여기서 얼마나 많은 보상이 남아 있는가?" → RL 이점에 대한 기준선

## 생성적 보상 모델링 (LLM-as-a-judge)

선호도 데이터의 비용으로 인해, 기존 언어 모델을 인간 선호도의 심판으로 또는 다른 평가 (evaluation) 설정에서 사용하는 대규모 연구 영역이 등장했다 [@zheng2023judging].
핵심 아이디어는 언어 모델에게 판단 방법에 대한 지시, 프롬프트, 그리고 두 개의 완성을 제공하는 것이다 (인간 레이블러에게 하는 것과 유사하게).
채팅 평가 MT-Bench [@zheng2023judging]를 위한 획기적인 연구 중 하나에서 가져온 예시 프롬프트는 다음과 같다:

```text
[System]
Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user question displayed below.
You should choose the assistant that follows the user's instructions and answers the user's question better.
Your evaluation should consider factors such as the helpfulness, relevance, accuracy, depth, creativity, and level of detail of their responses.
Begin your evaluation by comparing the two responses and provide a short explanation.
Avoid any position biases and ensure that the order in which the responses were presented does not influence your decision.
Do not allow the length of the responses to influence your evaluation.
Do not favor certain names of the assistants.
Be as objective as possible.
After providing your explanation, output your final verdict by strictly following this format: "[[A]]" if assistant A is better, "[[B]]" if assistant B is better, and "[[C]]" for a tie.
[User Question]
{question}
[The Start of Assistant A's Answer]
{answer_a}
[The End of Assistant A's Answer]
[The Start of Assistant B's Answer]
{answer_b}
[The End of Assistant B's Answer]
```

평가를 위한 LLM-as-a-judge의 효능으로 인해 AlpacaEval [@dubois2024length], Arena-Hard [@li2024crowdsourced], WildBench [@lin2024wildbench]와 같은 많은 다른 평가들이 생겨났으며, 많은 이들이 보상 모델 대신 선호도 데이터를 만들고 사용하기 위해 LLM-as-a-judge를 사용하기 시작했다.

"생성적 보상 모델 (Generative Reward Model)" [@mahan2024generative] [@zhang2024generative] [@ankner2024critique]을 어떻게 사용할지에 관한 연구 분야 전체가 등장했으며 (효과적인 심판으로 구체적으로 훈련된 모델 포함 [@kim2023prometheus]), 하지만 RM 평가에서는 기존 보상 모델에 뒤처지는 경향이 있어, 보상 모델링이 현재 RLHF에서 중요한 기술임을 보여 준다.

LLM-as-a-judge 워크플로의 견고성을 개선하는 일반적인 트릭은 평가 분산을 줄이기 위해 샘플링 온도 0을 사용하는 것이다.

## 더 읽어보기

보상 모델링에 대한 학술 문헌은 2024년에 자리를 잡았다.
보상 모델링의 초기 발전 대부분은 벤치마크 (benchmark) 구축과 행동 모드 파악에 초점을 맞추었다.
최초의 RM 벤치마크인 RewardBench는 보상 모델 테스트를 위한 공통 인프라를 제공했다 [@lambert2024rewardbench].
그 이후로 RM 평가는 일반적인 후처리 학습된 모델에 사용 가능한 평가 유형과 유사하게 확장되었으며, 일부 평가는 알려진 정답이 있는 도메인에서의 예측 정확도를 테스트하고 [@lambert2024rewardbench], 다른 일부는 LLM-as-a-judge로 수행되는 "느낌"이나 다른 벤치마크와의 상관관계에 더 가깝다 [@wen2024rethinking].

새로운 벤치마크의 예시는 다음과 같다:

- **텍스트 전용 (일반 채팅 / 선호도):** RMB [@zhou2024rmb], RewardBench2 [@malik2025rewardbench], Preference Proxy Evaluations [@frick2024evaluate], 또는 RM-Bench [@liu2024rm].
- **특화된 텍스트 전용 (수학 등):** 다국어 보상 벤치마크 (M-RewardBench) [@gureja2024m], 검색 증강 생성 (RAG, retrieval augmented generation)을 위한 RAG-RewardBench [@jin2024rag], 오타를 위한 ReWordBench [@wu2025rewordbench], RewardMATH [@kim2024evaluating], 또는 AceMath-RewardBench [@liu2024acemath].
- **과정 RM:** PRM Bench [@song2025prmbench] 또는 ProcessBench [@zheng2024processbench] 그리고 VisualProcessBench [@wang2025visualprm] 또는 ViLBench [@tu2025vilbench]의 시각적 벤치마크.
- **에이전틱 RM:** Agent-RewardBench [@men2025agentrewardbench] 또는 CUARewardBench [@lin2025cuarewardbench].
- **멀티모달:** MJ-Bench [@chen2024mj], Multimodal RewardBench [@yasunaga2025multimodal], VL RewardBench [@li2024vlrewardbench], 또는 VLRMBench [@ruan2025vlrmbench].

보상 모델 *훈련* 의 진전을 이해하려면, 측면 조건부 모델 (aspect-conditioned model) [@wang2024interpretable], 고품질 인간 데이터셋 [@wang2024helpsteer2] [@wang2024helpsteer2p], 스케일링 실험 [@adler2024nemotron], 광범위한 실험 [@touvron2023llama], 또는 데이터 편향 제거 [@park2024offsetbias]를 포함한 새로운 보상 모델 훈련 방법들을 참조할 수 있다.
