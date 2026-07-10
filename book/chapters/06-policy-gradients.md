<!--
  Copyright (c) 2025-2026 Nathan Lambert.
  Licensed under CC BY-NC-SA 4.0:
  https://creativecommons.org/licenses/by-nc-sa/4.0/
  Full license: https://github.com/natolambert/rlhf-book/blob/main/LICENSE-CHAPTERS
-->
---
prev-chapter: "보상 모델링"
prev-url: "05-reward-models"
page-title: 강화학습
search-title: "6장: 강화학습"
meta-description: "PPO, REINFORCE, RLOO, GRPO와 구현 세부사항을 포함해 RLHF와 LLM 사후 학습에 쓰이는 정책 그래디언트 방법을 설명합니다."
next-chapter: "추론과 추론 시간 스케일링"
next-url: "07-reasoning"
lectures:
  - video: "https://www.youtube.com/watch?v=K_Sj_-1BUMM&list=PLL1tdVxB1CpVpEtMHxwuR4uI4Lxjw00_y&index=4"
    label: "강의 3: LLM에서의 RL을 위한 정책 그래디언트 알고리즘 이해"
  - video: "https://www.youtube.com/watch?v=i-AIMpZHgeg&list=PLL1tdVxB1CpVpEtMHxwuR4uI4Lxjw00_y&index=5"
    label: "강의 4: LLM용 RL 알고리즘 구현"
---

# 강화학습 (즉, 정책 그래디언트 알고리즘)

RLHF 과정에서 강화학습 (RL) 알고리즘은 보상 모델로부터 받은 피드백을 기반으로 모델의 가중치를 점진적으로 업데이트한다.
정책(훈련 중인 모델)은 훈련 데이터셋의 프롬프트에 대한 완성을 생성하고, 보상 모델이 이를 점수화하면, 강화학습 (RL) 옵티마이저가 이 정보를 바탕으로 그래디언트 스텝을 수행한다 (개요는 @fig:rlhf-overview 참조).
이 장에서는 보상 모델이 온-정책 데이터에 제공하는 신호로부터 학습하는 데 사용되는 다양한 알고리즘의 수학적 원리와 트레이드오프를 설명한다.
이 알고리즘들은 수천 또는 수백만 개의 배치에 걸쳐 더 큰 프롬프트 집합에 대해 여러 에폭 동안 실행되며, 각 배치 사이마다 그래디언트 업데이트가 이루어진다.

RLHF를 언어 모델에 적용하면서 대중화된 알고리즘은 정책 그래디언트 강화학습 (RL) 알고리즘이다.
Proximal Policy Optimization (PPO, 근위 정책 최적화), Group Relative Policy Optimization (GRPO, 그룹 상대 정책 최적화), REINFORCE 등의 이 알고리즘들은 최근에 생성된 샘플을 사용하여 모델을 업데이트한다 (AlphaGo와 같은 프로젝트에서 사용된 Deep Q-Networks, DQN처럼 점수를 리플레이 버퍼에 저장하는 알고리즘과 달리).
이 절에서는 정책 그래디언트 알고리즘의 기초와 현대 RLHF 프레임워크에서의 활용 방식을 다룬다.

머신러닝 수준에서, 이 절의 내용은 RLHF 과정 중 가장 복잡한 주제에 해당한다.
그러나 대부분의 현대 AI 모델과 마찬가지로, 성공의 가장 큰 결정 요인은 해당 과정에 입력으로 제공되는 데이터이다.

![RLHF 훈련 루프 개요. 데이터셋의 프롬프트가 튜닝된 정책에 전달되면, 정책은 완성을 생성한다. 보상 모델이 이 완성에 점수를 매기는 동안, 고정된 초기 모델(일반적으로 RL 이전의 명령어 튜닝 모델)은 동일한 텍스트에 대한 로그 확률을 계산하여 과도한 이탈을 방지하는 KL 페널티를 산출한다. 결합된 보상 신호는 이후 정책 파라미터에 대한 강화학습 업데이트를 유도한다.](images/rlhf-overview.png){#fig:rlhf-overview}

ChatGPT와 함께 RLHF가 등장했을 때, PPO의 변형이 사용되었다는 것이 널리 알려져 있었고, 많은 초기 연구들이 그것을 기반으로 구축되었다.
시간이 지남에 따라 여러 연구 프로젝트들이 REINFORCE 스타일 알고리즘의 가능성을 보여주었으며 [@ahmadian2024back] [@wang2024helpsteer2p], 보상 모델 없이도 PPO에 비해 단순하고(메모리를 절약하여 필요한 GPU 수를 줄임), 가치 추정이 더 간단하다는 점(분산 감소를 위해 이점을 계산하는 방법인 Generalized Advantage Estimation, GAE 불필요)이 장점으로 부각되었다.
GRPO를 포함한 더 많은 알고리즘들이 등장했으며, 특히 추론 작업에서 인기를 끌고 있지만, 일반적으로 이러한 알고리즘 대부분은 특정 작업에 맞게 조정될 수 있다.
이 장에서는 정규 RLHF 문헌의 확립에 핵심적인 역할을 한 세 가지 알고리즘과 핵심 정책 그래디언트 설정을 다룬다.

가장 단순한 형태에서, RLHF의 RL 단계에는 두 가지 모델이 필요하다: 정책(훈련되는 모델)과 그 출력에 점수를 매기는 보상 모델(이전 장에서 다룸).
RL 이전의 정책 사본은 KL 페널티를 계산하기 위한 참조 모델로 사용된다(이 모델은 고정되어 있으며, 즉 자동 미분 엔진의 그래디언트로 업데이트되지 않는다).
여기서 다루는 가장 복잡한 알고리즘인 PPO는 네 번째 모델을 추가하는데, 이는 행동의 각 토큰이 얼마나 좋은지를 추정하는 데 사용되는 학습된 가치 함수로, 훈련 중에 업데이트되는 대규모 언어 모델 (LLM)이기도 하다.
이 장의 알고리즘들은 주로 *이점(advantages)*이라는 양을 추정하는 방법, 즉 모델의 현재 행동(완성)이 평균 대비 얼마나 좋은지를 측정하는 방식과, 최적화가 수치적으로 안정적이도록 정책 업데이트를 제한하는 방법에서 차이가 있다.
(가치 모델 없이) 이 RLHF 과정의 시각적 개요는 @fig:rlhf-overview 에 나와 있다.

기호 정의는 문제 설정 장을 참조하라.

*이 장에서는 강화학습 (RL) 문헌에서 사용하는 $(s, a)$ 표기법을 사용한다. 여기서 $s$는 상태, $a$는 행동을 나타낸다. 언어 모델 맥락에서는 흔히 $(x, y)$를 사용하는데, $x$는 프롬프트이고 $y$는 완성이다. $(s, a)$ 표현이 더 일반적이며, 이 알고리즘들은 각 타임스텝에서 행동이 취해지는 순차적 의사결정 문제를 위해 설계되었다. 그러나 많은 RLHF 구현에서는 전체 완성을 단일 행동으로 취급하므로, $(x, y)$ 표기법도 동등하게 유효하다.*

***RL 요약 참조표:** 이 장의 모든 핵심 RL 손실 함수를 한 페이지로 정리한 참조표는 [rlhfbook.com/rl-cheatsheet](https://rlhfbook.com/rl-cheatsheet) 에서 확인할 수 있다.*

## 정책 그래디언트 알고리즘

이 장의 핵심은 다음과 같은 형태의 수식을 이해하는 것이다.
이 수식은 우리가 훈련하는 언어 모델 $\pi_\theta$에 대한 그래디언트 $\Delta \theta$를 계산한다:

$$\Delta \theta \propto \Psi_t \, \nabla_\theta \log \pi_\theta(a_t \mid s_t)$$ {#eq:policy_gradient_intuition}

이 수식은 두 가지 핵심 구성 요소로 이루어져 있다:
1. $\nabla_\theta \log \pi_\theta(a_t \mid s_t)$ — 파라미터 공간에서 행동 $a_t$가 더 일어날 가능성이 높아지게 만드는 방향.
2. $\Psi_t$ — 그것이 얼마나 좋았는가? 결과를 점수화하는 스칼라.

이 두 값을 곱하면, 정책 그래디언트 업데이트를 얻게 된다.
몇 가지 사항은 직관적인데, 예를 들어 $\Psi_t > 0$이면 $a_t$가 더 일어날 가능성이 높아지도록 파라미터를 업데이트하고, $\Psi_t < 0$이면 가능성이 낮아지게 만든다.
정책 그래디언트는 어떤 파라미터가 행동에 기여했는지, 그리고 미래에 그 행동을 더 혹은 덜 발생시켜야 하는지를 계산한다.
이 장의 나머지 부분은 이를 수행하는 다양한 방법과 LLM에서 작동시키기 위한 구체적인 기법들을 깊이 있게 다룬다.

이제 이를 좀 더 형식화해 보자.
강화학습 (RL) 알고리즘은 상태 $s \in \mathcal{S}$와 행동 $a \in \mathcal{A}$의 궤적에 걸쳐 미래의 할인된 보상을 최대화하도록 설계된다 (더 많은 표기법은 부록 A, 정의를 참조).
에이전트의 목표, 흔히 *리턴*이라 불리는 것은 주어진 시간 $t$에서 시작하는 할인된 보상의 합이다 ($\gamma\in [0,1]$은 근기 보상을 우선시하는 할인 계수):

$$G_t = r_t + \gamma r_{t+1} + \cdots = \sum_{k=0}^\infty \gamma^k r_{t+k}.$$ {#eq:return_definition}

리턴 정의는 다음과 같이 재귀적으로 쓸 수도 있다:
$$G_{t} = r_t + \gamma G_{t+1}.$$ {#eq:recursive_return}

이 리턴은 현재 상태가 주어졌을 때 추정된 미래 리턴인 가치 함수 $V(s)$를 학습하는 기초가 된다:

$$V(s) = \mathbb{E}\left[G_t \mid S_t = s \right].$$ {#eq:value_function}

모든 정책 그래디언트 알고리즘은 기대 리턴을 최대화하도록 정책 $\pi_\theta(a\mid s)$를 최적화한다. 이 목적함수는 유도된 가치 함수 $V^{\pi_\theta}(s)$를 사용하여 표현될 수 있다.

$d_0(s)$를 초기 상태 분포라 하자. 우리가 최대화하는 에피소드 목적함수는 다음과 같이 쓸 수 있다:
$$
J(\theta)
\;=\;
\sum_{s} d_0(s) V^{\pi_\theta}(s),
$$ {#eq:policy_objective}

유한 MDP에서 이는 가능한 시작 상태에 대한 합이지만, 실제로는 정확히 계산하지 않는다.
대신, 현재 정책에서 롤아웃을 샘플링하여 데이터로부터 추정한다.
RLHF에서 이는 일반적으로 데이터셋에서 프롬프트 $x_i$를 샘플링하고 완성 $y_i \sim \pi_\theta(\cdot\mid x_i)$를 생성하는 것을 의미한다.
$R(x_i, y_i)$가 해당 프롬프트-완성 쌍에 부여된 스칼라 시퀀스 수준 보상이라고 하자. $\tau_i$가 대응되는 에피소드라면 이는 궤적 보상 $R(\tau_i)$이다.
그다음 다음과 같은 경험적 평균을 취한다:

$$
\hat{J}(\theta) = \frac{1}{B}\sum_{i=1}^{B} R(x_i, y_i),
$$ {#eq:empirical_batch_estimate}

또는 단계별 보상을 포함하는 MDP 관점에서,

$$
\hat{J}(\theta) = \frac{1}{B}\sum_{i=1}^{B} \sum_{t=0}^{T_i} \gamma^t r_{i,t}.
$$ {#eq:empirical_mdp_estimate}

실제로 언어 모델에 대한 RLHF는 $\gamma = 1$ (할인 없음)로 설정하는데, 최적화 단위가 개별 토큰이 아닌 집합적 완성이기 때문이다 — 이 선택은 이 장 후반부의 MDP 대 밴딧 절에서 추가로 논의된다.

정책 그래디언트 알고리즘의 핵심은 현재 정책에 대한 유한 시간 기대 리턴의 그래디언트를 계산하는 것이다.
이 기대 리턴 $J$로부터, 학습률 $\alpha$를 사용하여 다음과 같이 파라미터 업데이트를 계산할 수 있다:

$$\theta \leftarrow \theta + \alpha \nabla_\theta J(\theta)$$ {#eq:policy_update}

핵심 구현 세부사항은 해당 그래디언트를 어떻게 계산하는가이다.

### 정책 그래디언트 유도

$p_\theta(\tau)$를 초기 상태 분포 $d_0$, 정책 $\pi_\theta$, 환경 전이 동역학이 유도하는 궤적 분포라고 하자. 이는 아래 @eq:trajectory_probability 에서 전개된다.
최대화하고자 하는 RL 목적함수를 다른 방식으로 표현하면 다음과 같다:
$$
J(\theta) = \mathbb{E}_{\tau \sim p_\theta} \left[ R(\tau) \right],
$$ {#eq:policy_objective_expectation}

여기서 $\tau = (s_0, a_0, s_1, a_1, \ldots)$는 궤적이고 $R(\tau) = \sum_{t=0}^\infty r_t$는 궤적의 총 보상이다. 또는 모든 가능한 궤적에 대한 적분으로 기댓값을 쓸 수 있다:
$$
J(\theta) = \int_\tau p_\theta (\tau) R(\tau) d\tau
$$ {#eq:policy_objective_integral}

궤적 확률을 다음과 같이 표현할 수 있음에 주목하라. 여기서 $\pi_\theta(a_t|s_t) p(s_{t+1}|s_t, a_t)$는 정책 확률과 하나의 상태-행동 쌍에서 다음 상태로 가는 환경 전이 확률을 결합한 항이다:
$$
p_\theta (\tau) = d_0(s_0) \prod_{t=0}^\infty \pi_\theta(a_t|s_t) p(s_{t+1}|s_t, a_t),
$$ {#eq:trajectory_probability}

목적함수(@eq:policy_objective_expectation)에 대해 정책 파라미터 $\theta$에 관한 그래디언트를 취하면:
$$
\nabla_\theta J(\theta) = \int_\tau \nabla_\theta p_\theta (\tau) R(\tau) d\tau
$$ {#eq:policy_gradient_integral}

[로그-도함수 트릭](https://andrewcharlesjones.github.io/journal/log-derivative.html)을 사용하면 적분의 그래디언트를 기댓값으로 다시 쓸 수 있다:
$$
\begin{aligned}
\nabla_\theta \log p_\theta(\tau) &= \frac{\nabla_\theta p_\theta(\tau)}{p_\theta(\tau)} &\text{(연쇄 법칙에서)} \\
\implies \nabla_\theta p_\theta(\tau) &= p_\theta(\tau) \nabla_\theta \log p_\theta(\tau) &\text{(재정렬)}
\end{aligned}
$$ {#eq:log_chain_rule}

이 로그-도함수 트릭을 사용하면:
$$
\begin{aligned}
\nabla_\theta J(\theta) &= \int_\tau \nabla_\theta p_\theta (\tau) R(\tau) d\tau \\
&= \int_\tau p_\theta (\tau) R(\tau) \nabla_\theta \log p_\theta (\tau) d\tau \\
&= \mathbb{E}_{\tau \sim p_\theta} \left[ R(\tau) \nabla_\theta \log p_\theta (\tau) \right]
\end{aligned}
$$ {#eq:policy_gradient_expectation}

마지막 단계는 궤적 분포 $p_\theta(\tau)$ 하에서의 기댓값 정의를 사용한다: 임의의 함수 $f$에 대해, $\mathbb{E}_{\tau \sim p_\theta}[f(\tau)] = \int_\tau f(\tau)\,p_\theta(\tau)\,d\tau$ (이산적인 경우에는 합).
기댓값으로 표현하면 Monte Carlo 롤아웃, 예를 들어 현재 정책이 유도한 궤적 $\tau_i \sim p_\theta$에 대해 $\frac{1}{B}\sum_{i=1}^{B} f(\tau_i)$로 근사할 수 있어 유용하다.

유도로 돌아와, 궤적의 로그 확률을 전개하면:

$$
\log p_\theta (\tau) = \log d_0(s_0) + \sum_{t=0}^\infty \log \pi_\theta(a_t|s_t) + \sum_{t=0}^\infty \log p(s_{t+1}|s_t, a_t)
$$ {#eq:trajectory_log_prob}

위의 그래디언트를 취하면:

- $\nabla_\theta \log d_0(s_0) = 0$ (초기 상태 분포는 $\theta$에 의존하지 않음)
- $\nabla_\theta \log p(s_{t+1}|s_t, a_t) = 0$ (환경 전이 동역학은 $\theta$에 의존하지 않음)
- $\nabla_\theta \log \pi_\theta(a_t|s_t)$만 살아남음

따라서 궤적의 로그 확률에 대한 그래디언트는 다음과 같이 단순화된다:
$$
\nabla_\theta \log p_\theta (\tau) = \sum_{t=0}^\infty \nabla_\theta \log \pi_\theta(a_t|s_t)
$$ {#eq:trajectory_log_grad}

잠깐 짚어보면, 이 수식에 도달하는 것은 구현의 핵심 지점이다.
여기까지 진행하면, 궤적 분포의 그래디언트가 언어 모델 정책 확률(즉, 우리가 훈련하는 모델이 제공하는 토큰 확률)의 그래디언트 합으로 줄어들 수 있음을 알 수 있다.
실제로 이는 정책 그래디언트 수식의 일반적인 형태로 귀결된다.
결국 손실에 로그 확률의 합이 나타나고, autodiff를 통해 그래디언트를 계산하게 된다.
다음과 같은 짧은 코드가 반복적으로 등장한다:

```python
seq_log_probs = (token_log_probs * completion_mask).sum(dim=-1)
loss = -(seq_log_probs * advantages).mean()
loss.backward()
```

이 패턴은 이 장 전체에 걸쳐 나타난다. 이제 형식적인 정책 그래디언트 수학으로 돌아가자.

@eq:policy_gradient_expectation 에 이를 다시 대입하면:
$$
\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim p_\theta} \left[ \sum_{t=0}^\infty R(\tau) \nabla_\theta \log \pi_\theta(a_t|s_t) \right]
$$ {#eq:policy_gradient_returns}

종종 사람들은 더 일반적인 정책 그래디언트 공식화를 사용한다:
$$
g = \nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim p_\theta} \left[ \sum_{t=0}^\infty \Psi_t \nabla_\theta \log \pi_\theta(a_t|s_t) \right]
$$ {#eq:general_gradient}

여기서 $\Psi_t$는 다음 중 하나가 될 수 있다 (보상은 종종 $\gamma$로 할인될 수도 있음). Schulman et al. 2015 [@schulman2015high]에서 채택된 분류법:

1. $R(\tau) = \sum_{t=0}^{\infty} r_t$: 궤적의 총 보상.
2. $\sum_{t'=t}^{\infty} r_{t'}$: 행동 $a_t$ 이후의 보상, 시간 $t$에서의 리턴 $G_t$라고도 함.
3. $\sum_{t'=t}^{\infty} r_{t'} - b(s_t)$: 이전 공식의 기준선 버전.
4. $Q^{\pi}(s_t, a_t)$: 상태-행동 가치 함수.
5. $A^{\pi}(s_t, a_t)$: 이점 함수, 정확히 계산될 경우 이론적으로 가능한 최저 분산을 산출.
6. $r_t + \gamma V^{\pi}(s_{t+1}) - V^{\pi}(s_t)$: 시간차 (TD) 잔차.

*기준선*은 정책 업데이트의 분산을 줄이는 데 사용되는 값이다 (자세한 내용은 아래 참조).

언어 모델에서는 이러한 개념 중 일부가 그다지 의미 있지 않다.
예를 들어, 결정론적 정책 $\pi$에 대해 상태 가치는 $V^{\pi}(s_t) = Q^{\pi}(s_t, \pi(s_t))$이고 (최적 가치 함수에 대해서는 $V^*(s_t)=\max_{a_t} Q^*(s_t,a_t)$). 확률적 정책에서 유사한 항등식은 $V^{\pi}(s_t) = \mathbb{E}_{a_t \sim \pi(\cdot\mid s_t)}\!\left[Q^{\pi}(s_t,a_t)\right]$이다.
벨만 방정식은 Q와 V를 연결한다: 일반적으로 $Q^\pi(s_t,a_t) = \mathbb{E}\!\left[r_t + \gamma V^\pi(s_{t+1}) \mid s_t, a_t\right]$이지만, 상태 전이가 결정론적인 언어 모델에서는 $Q(s_t,a_t) = r_t + \gamma V(s_{t+1})$로 단순화된다.
이점 함수는 행동 $a_t$가 평균에 비해 얼마나 더 나은지를 측정한다:

$$A(s_t,a_t) = Q(s_t,a_t) - V(s_t) = r_t + \gamma V(s_{t+1}) - V(s_t)$$ {#eq:advantage_trick}

이 최종 형태는 정확히 시간차 (TD) 잔차(위의 항목 6)에 해당한다 — 가치 함수의 예측과 실제로 발생한 것 사이의 차이를 측정하는 RL의 기본 양으로, 가치 함수 업데이트를 더 정확한 추정치로 이끈다. 실제로는 학습된 가치 함수 $\hat{V}$가 이 TD 오차를 통해 이점을 추정하는 데 사용된다.

### 기본 정책 그래디언트 (Vanilla Policy Gradient)

기본 정책 그래디언트 구현은 정책 파라미터에 대해 미분하여 위의 $J(\theta)$ 표현식을 최적화한다.
시간 $t$의 리턴에 대한 간단한 버전은 다음과 같다:

$$\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim p_\theta} \left[ \sum_{t=0}^T G_t \nabla_\theta \log \pi_\theta(a_t|s_t) \right]$$ {#eq:vanilla_policy_gradient}

기본 정책 그래디언트 알고리즘의 일반적인 문제는 그래디언트 업데이트의 높은 분산이며, 이는 여러 방법으로 완화될 수 있다.
높은 분산은 리턴 $G$를 환경에서의 종종 소규모 롤아웃 집합으로부터 추정하는 것에서 비롯되는데, 이는 노이즈에 취약한 경향이 있다 (예를 들어 온도 $>0$에서 언어 모델로 생성하는 것의 확률론적 특성).
보상이 희박한 도메인에서는 리턴 추정치의 분산이 더 높은데, 더 많은 샘플이 밀집된 클러스터가 아닌 0 또는 1이기 때문이다.
이를 완화하기 위해 가치 추정을 정규화하는 다양한 기법인 *기준선*이 사용된다.
기준선은 하류 행동 대비 상태의 가치로 정규화하는 등 여러 방식으로 이를 달성한다 (예를 들어 Q 가치와 가치의 차이인 이점의 경우).
가장 단순한 기준선은 보상 배치의 평균이나 이동 평균이다.
이러한 행동 독립적 기준선조차도 기대 그래디언트를 변경하지 않고 분산을 줄일 수 있다. 임의의 상태 의존 $b(s)$에 대해 $\mathbb{E}_{a \sim \pi(a|s)}\!\left[b(s) \nabla_\theta \log \pi_\theta(a|s)\right] = 0$이 성립하여 학습 신호를 크게 개선한다.

이 장에서 논의되는 많은 정책 그래디언트 알고리즘은 이점 공식화를 기반으로 한다:

$$\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim p_\theta} \left[ \sum_{t=0}^T A^{\pi_\theta}(s_t, a_t) \nabla_\theta \log \pi_\theta(a_t|s_t) \right]$$ {#eq:advantage_policy_gradient}


### REINFORCE

REINFORCE 알고리즘은 아마도 두문자어일 가능성이 높지만, 그것이 나타내는 알고리즘의 구성 요소들은 현대 강화학습 알고리즘에서도 매우 관련성이 높다.
*연결주의 강화학습을 위한 단순 통계적 그래디언트 추적 알고리즘 (Simple statistical gradient-following algorithms for connectionist reinforcement learning)* [@williams1992simple]이라는 기념비적 논문에서 정의되었다:

> 이름은 "REward Increment = Nonnegative Factor X Offset Reinforcement X Characteristic Eligibility"의 두문자어이다.

이 세 가지 구성 요소는 *보상 증가분*, 즉 정책 그래디언트 스텝을 어떻게 수행하는가를 나타낸다.
업데이트 규칙의 세 가지 부분이 있다:

1. 비음 계수 (Nonnegative factor): 양수여야 하는 학습률(스텝 크기), 예를 들어 아래의 $\alpha$.
2. 오프셋 강화 (Offset Reinforcement): 안정성을 향상시키기 위한 보상의 기준선 $b$ 또는 기타 정규화 인수.
3. 특성 적격성 (Characteristic Eligibility): 스칼라 보상 신호를 행동을 생성한 파라미터에 귀속시킨다. Williams는 이 적격성 항을 $e$로 나타낸다 (지수 함수가 아님). 현대 정책 그래디언트 표기법에서 이는 $\nabla_\theta \log \pi_\theta(a_t \mid s_t)$에 해당한다.

따라서 형태는 꽤 친숙해 보인다:

$$ \Delta_\theta = \alpha(r - b)e $$ {#eq:REINFORCE_BASIC}

더 현대적인 표기법과 일반화된 리턴 $G$를 사용하면 REINFORCE 연산자는 다음과 같이 나타난다:

$$
\nabla_{\theta}\,J(\theta)
\;=\;
\mathbb{E}_{\tau \sim p_{\theta}}\!\left[
    \sum_{t=0}^{T}
    (G_t - b(s_t))\,\nabla_{\theta} \log \pi_{\theta}(a_t \mid s_t)
\right],
$$ {#eq:REINFORCE_with_baseline}

여기서 값 $G_t - b(s_t)$는 현재 상태에서 정책의 *이점*이므로, 이점 $A$를 사용하여 정책 그래디언트를 이후에도 계속 사용하는 형태로 재공식화할 수 있다:

$$
\nabla_{\theta}\,J(\theta)
\;=\;
\mathbb{E}_{\tau \sim p_{\theta}}\!\left[
    \sum_{t=0}^{T}
    A_t\,\nabla_{\theta} \log \pi_{\theta}(a_t \mid s_t)
\right],
$$ {#eq:REINFORCE_with_advantage}

REINFORCE는 그래디언트의 Monte Carlo 추정기를 사용하는 기본 정책 그래디언트의 특정 구현이다.

![언어 모델을 위한 기본 REINFORCE 구조. 형성된 보상은 보상 모델 점수와 참조 모델의 KL 페널티를 결합한다. 이 구조를 이 장 전체에서 발전시켜 나간다.](images/reinforce_tikz.png){#fig:reinforce-arch data-dark-src="images/reinforce_tikz-dark.png"}

### REINFORCE Leave One Out (RLOO)

표준 REINFORCE 대비 REINFORCE Leave One Out (RLOO)의 핵심 구현 세부사항은, 기준선을 계산하기 위해 배치 내 모든 보상의 평균 대신 *다른* 샘플들의 평균 보상을 사용한다는 것이다 [@huang2024putting], [@ahmadian2024back], [@kool2019buy].
현재 샘플의 보상을 자체 기준선에서 제외함으로써, RLOO 기준선은 평가 중인 행동과 독립적이 되어 그래디언트 추정기가 정확히 불편향(unbiased)을 유지한다.

중요하게도, 이는 하나의 상태(프롬프트)당 여러 궤적(완성)을 생성할 때만 작동하며, 이는 RL로 언어 모델을 미세조정하는 여러 도메인에서 일반적인 관행이다.

구체적으로, REINFORCE Leave-One-Out (RLOO) 기준선에서, 주어진 프롬프트 $s$에 대한 $K$개의 샘플링된 궤적(프롬프트 조건 하에 취해진 행동) $a_1, \dots, a_K$에 대해 *프롬프트별* 기준선을 다음과 같이 명시적으로 정의한다:

$$
b(s, a_k) = \frac{1}{K-1}\sum_{i=1, i\neq k}^{K} R(s, a_i),
$$ {#eq:RLOO_baseline}

이로부터 다음의 이점이 도출된다:

$$
A(s, a_k) = R(s, a_k) - b(s, a_k).
$$ {#eq:RLOO_advantage}

동등하게, 다음과 같이 표현할 수도 있다:

$$
A(s, a_k) = \frac{K}{K - 1}\left(R(s, a_k) - \frac{1}{K}\sum_{i=1}^{K} R(s, a_i)\right).
$$ {#eq:RLOO_advantage_alt}

이는 GRPO (그룹 상대 정책 최적화, 곧 PPO 이후에 논의됨)에서 사용되는 그룹 상대 이점과 밀접하게 관련된 단순하고 저분산의 *프롬프트별* 이점 추정치이다.
실제로 GRPO 스타일 훈련은 주로 KL 정규화 인자를 어떻게 적용하는가(명시적 손실 항 대 보상에 포함)와 PPO 스타일 비율 클리핑 사용 여부에서 차이가 난다.
구체적으로, 표준 GRPO 구현은 손실 수준에서 KL 페널티를 적용하는 반면, RLOO나 전통적인 정책 그래디언트의 유도는 보상 자체에 KL 페널티를 적용한다.
RLHF에서 추론 및 검증 가능한 보상을 사용한 강화학습 (RLVR)으로의 전환에 따라 KL 페널티의 사용이 전반적으로 감소했으며, 많은 추론 기반 RLHF 코드 적용에서 KL 페널티를 완전히 끄는 경우도 있다.
그렇더라도 RLOO의 이점은 PPO의 클리핑과 결합될 수 있어, 이러한 알고리즘들이 얼마나 유사한지를 보여준다.

RLOO 및 가치 네트워크를 사용하지 않는 다른 알고리즘들 — 각 토큰당 스칼라 가치 $V(s_t)$를 예측하는 추가 모델 사본(비평자) — 은 손실을 계산할 때 모든 토큰에 동일한 시퀀스 수준의 이점(또는 보상)을 할당한다.
PPO와 같이 학습된 가치 네트워크를 사용하는 알고리즘은 EOS 토큰에서 달성된 최종 보상으로부터 할인하여 각 토큰에 개별적으로 다른 가치를 할당한다.
KL 거리 페널티를 사용하면, RLOO는 완성에 걸친 토큰별 KL을 집계하고 그 스칼라를 시퀀스 보상에 포함시켜, 결과적인 이점이 모든 토큰에 브로드캐스트된다.
PPO는 $A_t$를 계산하기 전에 토큰별 KL을 토큰별 보상에서 차감하여 토큰 수준의 크레딧 할당을 제공한다.
GRPO는 일반적으로 보상에서 차감하는 대신 손실에 별도의 토큰별 항을 추가하여 시퀀스 수준의 이점을 유지한다.
이러한 세부사항과 트레이드오프는 이 장 후반에서 논의된다.

![REINFORCE Leave-One-Out (RLOO) 구조. 프롬프트당 여러 완성이 가치 함수 학습 없이 이점 추정을 위한 leave-one-out 기준선을 제공한다.](images/rloo_tikz.png){#fig:rloo-arch data-dark-src="images/rloo_tikz-dark.png"}

<!-- A nice formulation of LM RL loss functions is found here https://arxiv.org/pdf/2502.01600 -->

### Proximal Policy Optimization (PPO, 근위 정책 최적화)

Proximal Policy Optimization (PPO) [@schulman2017proximal]은 Deep RL의 성공(예: Dota 2를 정복한 OpenAI Five [@berner2019dota] 및 방대한 양의 연구)의 기반이 되는 핵심 알고리즘 중 하나이다.
PPO가 이점과 정책 확률에 대해 최대화하는 목적함수는 다음과 같다:

$$J(\theta) = \min\left(\frac{\pi_\theta(a|s)}{\pi_{\theta_{\text{old}}}(a|s)}A, \text{clip} \left( \frac{\pi_\theta(a|s)}{\pi_{\theta_{\text{old}}}(a|s)}, 1-\varepsilon, 1+\varepsilon \right) A \right).$$ {#eq:PPO_EQN}

여기서 $\pi_\theta(a|s)$는 최적화 중인 현재 정책이고, $\pi_{\theta_{\text{old}}}(a|s)$는 훈련 데이터를 수집하는 데 사용된 정책(이전 반복의 정책)이다.
두 정책 간의 비율은 *중요도 샘플링*에서 나오는데, 이를 통해 새로운 정책에 대한 그래디언트를 추정하기 위해 이전 정책에서 수집된 데이터를 재사용할 수 있다.

정책 그래디언트의 이점 공식화(@eq:advantage_policy_gradient)를 다시 상기하면:
$$\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim p_\theta} \left[ \sum_{t=0}^T A^{\pi_\theta}(s_t, a_t) \nabla_\theta \log \pi_\theta(a_t|s_t) \right].$$ {#eq:advantage_policy_gradient_recall}

이 기댓값은 $\pi_\theta$가 유도한 궤적 분포에서 샘플링된 궤적에 대해 계산되지만, 실제로는 고정된 정책 $\pi_{\theta_{\text{old}}}$에서 수집된 데이터 배치에 대해 여러 그래디언트 스텝을 취하고 싶다.
이 분포 불일치를 보정하기 위해 중요도 가중치 $\frac{\pi_\theta(a|s)}{\pi_{\theta_{\text{old}}}(a|s)}$를 곱하는데, 이는 현재 정책 대비 데이터 수집 정책에서 샘플의 가능도가 얼마나 높거나 낮은지를 반영하여 샘플을 재가중한다.
제약 없이 이 중요도 가중 목적함수를 최적화하면 비율이 1에서 크게 벗어날 때 파괴적으로 큰 정책 업데이트가 발생할 수 있다.
PPO는 비율을 범위 $[1-\varepsilon, 1+\varepsilon]$으로 클리핑하여 단일 업데이트에서 정책이 너무 크게 변하지 않도록 보장함으로써 이를 해결한다.

PPO와 그 주변 알고리즘으로 넘어가면 명시적인 그래디언트보다 *목적함수*를 다루는 경우가 많다는 점에 유의하자.
이는 PPO 목적함수가 $\min$과 클리핑 연산을 포함하면 쉽게 해석되는 해석적 그래디언트를 갖지 않기 때문이다(@fig:ppo-obj 의 영역에 따라, 쓰는 방식에 따라 약 4개의 항이 생긴다). 따라서 목적함수를 쓰는 것이 이런 알고리즘을 전달하는 더 명확한 방식이다.

완전성을 위해, PPO는 일반적으로 타임스텝에 걸친 *기대* 클리핑된 대리 목적함수로 작성된다:

$$
J(\theta)
=
\mathbb{E}_{t}\left[
\min\left(\rho_t(\theta)A_t,\ \text{clip}(\rho_t(\theta),1-\varepsilon,1+\varepsilon)A_t\right)
\right],
\qquad
\rho_t(\theta)=\frac{\pi_\theta(a_t\mid s_t)}{\pi_{\theta_{\text{old}}}(a_t\mid s_t)}.
$$ {#eq:PPO_EQN_EXPECTED}

목적함수는 종종 음수 부호를 추가하여 손실 함수로 변환되며, 이를 통해 옵티마이저는 가능한 한 음수로 만들려 한다.

언어 모델의 경우, 목적함수(또는 손실)는 토큰별로 계산된다. 이는 자기회귀 예측의 전체 시퀀스 확률을 확률의 곱으로 계산하는 방식에 직관적으로 기반한다.
여기서 일반적인 구현은 현대 언어 모델링 프레임워크에서 더 단순하게 계산할 수 있는 *로그 확률*을 사용한다.
실제로 토큰 로그 확률의 차이를 계산하고 지수화하여 정책 비율 $\rho_t$를 복원한다.

$$ J(\theta) = \frac{1}{|a|} \sum_{t=0}^{|a|} \min\left(\frac{\pi_\theta(a_{t}|s_t)}{\pi_{\theta_{\text{old}}}(a_{t}|s_t)}A_{t}, \text{clip} \left( \frac{\pi_\theta(a_{t}|s_t)}{\pi_{\theta_{\text{old}}}(a_{t}|s_t)}, 1-\varepsilon, 1+\varepsilon \right) A_{t} \right).  $$  {#eq:PPO_EQN_EXPANDED}

이것은 다른 정책 그래디언트 방법에도 적용되는 PPO의 토큰별 버전으로, 이 장의 구현 절에서 더 자세히 탐구된다.
여기서 행동의 토큰 수로 평균을 내는 항 $\frac{1}{|a|}$는 일반적인 구현 관행에서 비롯되지만, 손실의 공식적인 유도에는 없다 ([@liu2025understanding]에서 논의됨).

![PPO 프레임워크. 학습된 가치 함수는 클리핑된 대리 목적함수와 함께 사용되는 토큰별 이점을 위한 Generalized Advantage Estimation (GAE)를 가능하게 한다.](images/ppo_tikz.png){#fig:ppo-arch data-dark-src="images/ppo_tikz-dark.png"}

여기서는 다양한 이점과 정책 비율에 대해 이 손실 함수가 유발하는 다양한 경우를 설명한다.
구현 수준에서, PPO의 내부 계산은 두 가지 주요 항을 포함한다: 1) 학습된 이점을 사용한 표준 정책 그래디언트와 2) 최대 스텝 크기에 기반한 클리핑된 정책 그래디언트.

다양한 상황이 어떻게 발생하는지 이해하기 위해 정책 비율을 다음과 같이 정의한다:

$$\rho(\theta) = \frac{\pi_\theta(a|s)}{\pi_{\theta_{\text{old}}}(a|s)}$$ {#eq:PPO_POL_RATIO}

정책 비율은 PPO 및 관련 알고리즘의 핵심이다.
정책의 그래디언트를 계산하는 과정에서 나타나며, 매우 직관적인 방식으로 파라미터 업데이트를 제어한다.
임의의 데이터 배치에서, 정책 비율은 이 배치에 대한 첫 번째 그래디언트 스텝에서 1로 시작한다. 이 시점에서 $\pi_{\theta}$와 $\pi_{\theta_{\text{old}}}$가 동일하기 때문이다. 그런 다음 다음 그래디언트 스텝에서, 해당 그래디언트 스텝이 긍정적인 이점을 가진 특정 토큰의 likelihood를 증가시켰다면 정책 비율은 1보다 커지고, 반대의 경우에는 1보다 작아진다. 일반적인 관행은 $\pi_{\theta_{\text{old}}}$를 업데이트하기 전에 배치당 1-4 그래디언트 스텝을 취하는 것이다.

#### PPO 목적함수 이해

전반적으로, PPO 목적함수는 목적함수 대 정책 비율의 플롯의 두 선으로 시각화할 수 있으며, 이는 @fig:ppo-obj 에 나와 있다.
PPO 목적함수는 샘플링된 행동의 확률을 변경하여 최대화된다.
수치적으로, 목적함수는 최솟값 연산의 영리한 사용을 통해 긍정적 이점과 부정적 이점 모두를 제어하여, 업데이트가 정책 비율 1로부터 최대 엡실론 거리만큼만 이동하도록 한다.

신뢰 영역 내에서 PPO는 다른 정책 그래디언트 알고리즘과 동일하게 작동한다.
이것은 의도적인 설계이다! 신뢰 영역은 안정적인 업데이트를 위해 PPO와 유사 알고리즘의 최대 스텝 크기를 제한하는 데 사용되는 개념이다. PPO 알고리즘의 핵심인 클립 및 min/max 함수는 이 영역을 정의한다. 목적함수는 그 외부에서는 평탄해진다.

"신뢰 영역"의 개념은 수치 최적화 문헌에서 유래하였지만 [@nocedal2006numerical], Deep RL 내에서는 PPO의 전신으로 인정받는 알고리즘 Trust Region Policy Optimization (TRPO)으로부터 대중화되었다 [@schulman2015trust].
신뢰 영역은 PPO 목적함수의 max/min 연산에 의해 업데이트가 "클리핑"되지 않는 완전한 정책 그래디언트 스텝이 적용되는 영역이다.

![양의 이점과 음의 이점 모두에 대해 정책 비율 $\rho(\theta)$의 함수로 본 PPO 목적함수 $J(\theta)$ 시각화. 각 패널은 세 비율 영역에 대해 비클리핑 항, 클리핑 항, 결과 목적함수, 그래디언트를 표시한다.](images/ppo-clip-viz.png){#fig:ppo-obj}

정책 비율과 이점은 몇 가지 다른 구성으로 발생할 수 있으며, @fig:ppo-obj 는 이점 $A_t$의 부호와 정책 비율 $\rho(\theta)$가 속한 세 영역에 따라 이를 열거한다. 모든 영역의 결과는 두 가지 사실로 결정된다. 이점의 부호는 해당 행동을 더 가능하게 만들고 싶은지 덜 가능하게 만들고 싶은지를 정하고, $\min$ 연산은 비클리핑 항 $\rho(\theta) A_t$ 또는 그 클리핑된 대응항 중 하나를 선택한다.

클리핑은 정책이 이미 샘플링된 행동을 원하는 방향으로 신뢰 영역의 경계를 넘어 충분히 이동시킨 두 영역에서만 그래디언트를 0으로 만든다:

- **양의 이점이고 $\rho(\theta) > 1+\varepsilon$인 경우**: 해당 행동은 이미 $\pi_{\theta_{\text{old}}}$보다 $\pi_\theta$에서 상당히 더 가능해졌다. 목적함수는 $(1+\varepsilon)A_t$에서 포화되고, 그래디언트는 0이며, 더 이상 업데이트하지 않는다. 이미 충분히 강화된 행동을 과도하게 더 강화하지 않기 위함이다.
- **음의 이점이고 $\rho(\theta) < 1-\varepsilon$인 경우**: 해당 행동은 이미 $\pi_\theta$에서 상당히 덜 가능해졌다. 목적함수는 $(1-\varepsilon)A_t$에서 포화되고, 그래디언트 역시 0이며, 더 이상 업데이트하지 않는다. 이미 억제된 행동을 과도하게 더 억제하지 않기 위함이다.

그 밖의 모든 곳에서는 비클리핑 항 $\rho(\theta) A_t$가 활성화되어 PPO가 표준 정책 그래디언트 스텝을 수행한다. 즉 $A_t > 0$이면 해당 행동의 확률을 높이고, $A_t < 0$이면 낮춘다. @fig:ppo-obj 의 각 영역은 업데이트된 정책 $\pi_\theta$가 무엇을 하도록 요구받는지로 읽을 수 있다:

- 양의 이점에서 기울어진 비클리핑 영역(초록색)은 샘플링된 행동의 확률을 **증가**시킨다.
- 음의 이점에서 기울어진 비클리핑 영역(빨간색)은 그 확률을 **감소**시킨다.
- 평평한 클리핑 영역(회색)은 그래디언트가 0이므로 정책을 **변경하지 않는다**.

같은 영역을 항별로 쓰면 다음과 같다:

**긍정적 이점 ($A_t > 0$)**

이는 취해진 행동이 가치 함수에 따르면 유익했음을 의미하며, 미래에 그 행동을 취할 가능도를 높이고 싶다는 것이다. 이제 정책 비율 $\rho(\theta)$의 다양한 경우를 살펴보자:

1. $\rho(\theta) < 1 - \varepsilon$:

    - **해석**: 새 정책에서 행동의 가능도가 이전 정책보다 낮음
    - **비클리핑 항**: $\rho(\theta) A_t$
    - **클리핑 항**: $(1 - \varepsilon) A_t$
    - **목적함수**: $\rho(\theta) A_t$
    - **그래디언트**: $\nabla_\theta \rho(\theta) A_t \neq 0$
    - **결과**: 일반 정책 그래디언트 업데이트 - 행동의 가능도 증가

2. $1 - \varepsilon \leq \rho(\theta) \leq 1 + \varepsilon$:

    - **해석**: 새 정책에서 행동의 가능도가 이전 정책과 거의 동일함
    - **비클리핑 항**: $\rho(\theta) A_t$
    - **클리핑 항**: $\rho(\theta) A_t$
    - **목적함수**: $\rho(\theta) A_t$
    - **그래디언트**: $\nabla_\theta \rho(\theta) A_t \neq 0$
    - **결과**: 일반 정책 그래디언트 업데이트 - 행동의 가능도 증가

3. $1 + \varepsilon < \rho(\theta)$:

    - **해석**: 새 정책에서 행동의 가능도가 이전 정책보다 높음
    - **비클리핑 항**: $\rho(\theta) A_t$
    - **클리핑 항**: $(1 + \varepsilon) A_t$
    - **목적함수**: $(1 + \varepsilon) A_t$
    - **그래디언트**: $\nabla_\theta (1 + \varepsilon) A_t = 0$
    - **결과**: 업데이트 없음 - 새 정책에서 행동의 가능도가 이미 높음

요약하면, 이점이 긍정적($A_t>0$)일 때 행동의 확률을 높이고 싶다. 따라서:

- $\pi_{\text{new}}(a) \leq (1+\varepsilon) \pi_{\text{old}}(a)$인 경우에만 그래디언트 스텝을 수행한다. 직관적으로, 이점이 긍정적이었으므로 행동의 확률을 높이고 싶지만, 과도하게 가능도를 높이고 싶지는 않다.
- 결정적으로, $\pi_{\text{new}}(a) > (1+\varepsilon) \pi_{\text{old}}(a)$일 때는 어떤 업데이트도 수행하지 않으며, 클리핑된 목적함수의 그래디언트는 $0$이다. 직관적으로, 새 정책에서 행동이 이미 더 많이 표현되어 있으므로 과도하게 강화하고 싶지 않다.

**부정적 이점 ($A_t < 0$)**

이는 취해진 행동이 가치 함수에 따르면 해로웠음을 의미하며, 미래에 그 행동을 취할 가능도를 낮추고 싶다는 것이다. 이제 정책 비율 $\rho(\theta)$의 다양한 경우를 살펴보자:

1. $\rho(\theta) < 1 - \varepsilon$:

    - **해석**: 새 정책에서 행동의 가능도가 이전 정책보다 낮음
    - **비클리핑 항**: $\rho(\theta) A_t$
    - **클리핑 항**: $(1 - \varepsilon) A_t$
    - **목적함수**: $(1 - \varepsilon) A_t$
    - **그래디언트**: $\nabla_\theta (1 - \varepsilon) A_t = 0$
    - **결과**: 업데이트 없음 - 새 정책에서 행동의 가능도가 이미 낮음

2. $1 - \varepsilon \leq \rho(\theta) \leq 1 + \varepsilon$:

    - **해석**: 새 정책에서 행동의 가능도가 이전 정책과 거의 동일함
    - **비클리핑 항**: $\rho(\theta) A_t$
    - **클리핑 항**: $\rho(\theta) A_t$
    - **목적함수**: $\rho(\theta) A_t$
    - **그래디언트**: $\nabla_\theta \rho(\theta) A_t \neq 0$
    - **결과**: 일반 정책 그래디언트 업데이트 - 행동의 가능도 감소

3. $1 + \varepsilon < \rho(\theta)$:

    - **해석**: 새 정책에서 행동의 가능도가 이전 정책보다 높음
    - **비클리핑 항**: $\rho(\theta) A_t$
    - **클리핑 항**: $(1 + \varepsilon) A_t$
    - **목적함수**: $\rho(\theta) A_t$
    - **그래디언트**: $\nabla_\theta \rho(\theta) A_t \neq 0$
    - **결과**: 일반 정책 그래디언트 업데이트 - 행동의 가능도 감소

요약하면, 이점이 부정적($A_t < 0$)일 때 행동의 확률을 낮추고 싶다. 따라서:

- $\pi_{\text{new}}(a) \geq (1-\varepsilon) \pi_{\text{old}}(a)$인 경우에만 그래디언트 스텝을 수행한다. 직관적으로, 이점이 부정적이었으므로 행동의 확률을 낮추고 싶으며, 이점에 비례하여 그렇게 한다.
- 결정적으로, $\pi_{\text{new}}(a) < (1-\varepsilon) \pi_{\text{old}}(a)$일 때는 어떤 업데이트도 수행하지 않으며, 클리핑된 목적함수의 그래디언트는 $0$이다. 직관적으로, 새 정책에서 행동의 가능도가 이미 낮으므로 과도하게 억제하고 싶지 않다.

신뢰 영역 내에서 PPO는 표준 정책 그래디언트 형태와 거의 동일하다는 점을 기억하는 것이 중요하다.


#### 가치 함수와 PPO

PPO 내의 가치 함수는 토큰별 가치를 예측하는 데 사용되는 모델의 추가 사본이다.
전통적인 RL에서 토큰(또는 상태)의 가치는 해당 시점부터의 미래 리턴을 예측하는 것으로, 종종 할인을 포함한다.
PPO에서 이 가치는 학습된 기준선으로 사용되며, REINFORCE에서 사용되는 단순한 Monte Carlo 버전(학습된 가치 네트워크가 필요하지 않음)의 발전형이다.
이는 PPO가 최적화 형태, 기준선 등 여러 측면에서 REINFORCE 및 기본 정책 그래디언트의 발전임을 잘 보여준다.
실제로 언어 모델에 사용되는 PPO 및 다른 알고리즘에서, 이는 KL 페널티 공제 후 각 토큰의 리턴을 예측하는 것이다 (전통적으로 토큰별 손실에 KL이 보상으로 포함된 것처럼, 앞서 논의했다).

가치 함수를 학습하는 데 사용되는 몇 가지 다른 방법(또는 목표)이 있다.
Generalized Advantage Estimation (GAE)은 현대 시스템에서 최신이자 정식 구현으로 간주되지만, 여러 스텝에 걸쳐 가치 예측 오차를 계산함으로써 더 높은 복잡성을 수반한다 — 이 장 후반의 GAE 절을 참조하라.
가치 함수는 정책을 업데이트하는 데 사용된 롤아웃의 Monte Carlo 추정치로도 학습될 수 있다.
PPO는 두 가지 손실을 가진다 — 하나는 가치 함수를 학습하고 다른 하나는 그 가치 함수를 사용하여 정책을 업데이트한다.

![가치 함수 훈련은 온-정책 롤아웃을 사용하여 목표를 계산한다. 모델은 각 토큰에서 $V_t$를 예측하고, 이는 목표 리턴 $\hat{V}_t$에 대해 MSE로 훈련된다. 그러면 이점 $A_t = \hat{V}_t - V_t$가 정책 그래디언트 업데이트에 가중치를 부여한다.](images/value_fn_training.png){#fig:value_fn_training data-dark-src="images/value_fn_training-dark.png"}

아래에 가치 네트워크 손실의 간단한 예제 구현이 나와 있다.

```python
# Basic PPO critic targets & loss (no GAE)
#
# B: Batch Size
# L: Completion Length
# Inputs:
#   rewards: (B, L) post-KL per-token rewards; EOS row includes outcome
#   done_mask: (B, L) 1.0 at terminal token (EOS or truncation if penalized), else 0.0
#   completion_mask: (B, L) 1.0 on response tokens to supervise (ignore the prompt)
#   values: (B, L) current critic predictions V_theta(s_t)
#       because a value network is a running update
#   old_values: (B, L) critic predictions at rollout time V_{theta_old}(s_t)
#   gamma: discount factor, float (often 1.0 for LM RLHF)
#   epsilon_v: float value clip range (e.g., 0.2), similar to PPO Loss Update itself, optional
#
# Returns:
#   value_loss: scalar; advantages: (B, L) detached (for policy loss)

B, L = rewards.shape

# 1) Monte Carlo returns per token (reset at terminals)
# Apply discounting, if enabled
returns = torch.zeros_like(rewards)
running = torch.zeros(B, device=rewards.device, dtype=rewards.dtype)
for t in reversed(range(L)):
    running = rewards[:, t] + gamma * (1.0 - done_mask[:, t]) * running
    returns[:, t] = running

targets = returns  # y_t = G_t (post-KL)

# 2) PPO-style value clipping (optional)
v_pred = values
v_old  = old_values
v_clip = torch.clamp(v_pred, v_old - epsilon_v, v_old + epsilon_v)

vf_unclipped = 0.5 * (v_pred - targets) ** 2
vf_clipped   = 0.5 * (v_clip - targets) ** 2
vf_loss_tok  = torch.max(vf_unclipped, vf_clipped)

# 3) Mask to response tokens and aggregate
denom = completion_mask.sum(dim=1).clamp_min(1)
value_loss = ((vf_loss_tok * completion_mask).sum(dim=1) / denom).mean()

# 4) Advantages for policy loss (no GAE): A_t = G_t - V(s_t)
advantages = (targets - v_pred).detach()

# The value loss is applied later, often with the PG loss, e.g.
# total_loss = policy_loss + vf_coef * value_loss
```

### Group Relative Policy Optimization (GRPO, 그룹 상대 정책 최적화)

Group Relative Policy Optimization (GRPO)은 DeepSeekMath [@shao2024deepseekmath]에서 도입되었고, DeepSeek-V3 [@deepseekai2025deepseekv3technicalreport] 및 DeepSeek-R1 [@guo2025deepseek] 등의 다른 DeepSeek 연구에서도 사용된다.
GRPO는 매우 유사한 대리 손실을 가진 PPO 기반 알고리즘으로 볼 수 있지만, 원래 정책 언어 모델(또는 초기화를 위한 다른 체크포인트)의 또 다른 사본으로 가치 함수를 학습하는 것을 피한다.
이를 통해 두 가지 장점이 있다고 제안된다:

1. LM 백본에서 가치 함수를 학습하는 어려움 회피 (연구에서 모범 사례가 아직 확립되지 않음).
2. 메모리에 추가 모델 가중치를 유지할 필요가 없어 메모리 절약 (현재 정책, 참조 정책, 가치 함수를 모두 유지해야 하는 것에서 처음 두 사본만으로 줄어듦).

GRPO는 여러 완성($a_i$)과 보상($r_i$)을, 즉 Monte Carlo 추정치를 동일한 초기 상태/프롬프트($s$)에서 수집하여 이점 또는 기준선을 추정함으로써 가치 추정을 단순화하고 에피소드의 모든 토큰(즉, 프롬프트에 대한 완성에서 각 토큰은 표준 가치 함수에서의 할인된 보상 대신 동일한 가치를 부여받음)에 동일한 가치를 할당한다.

이를 형식적으로 표현하면, GRPO 목적함수는 위의 PPO 목적함수와 매우 유사하다.
GRPO의 경우, 목적함수(또는 손실)는 주어진 프롬프트 $s$에 대한 완성 그룹 $\{a_1, a_2, ..., a_G\}$에 걸쳐 누적된다.
다음은 GRPO 목적함수이다:

$$J(\theta) = \frac{1}{G}\sum_{i=1}^G \left(\min\left(\frac{\pi_\theta(a_i|s)}{\pi_{\theta_{\text{old}}}(a_i|s)}A_i, \text{clip} \left( \frac{\pi_\theta(a_i|s)}{\pi_{\theta_{\text{old}}}(a_i|s)}, 1-\varepsilon, 1+\varepsilon \right) A_i \right) - \beta \mathcal{D}_{\text{KL}}(\pi_\theta||\pi_{\text{ref}})\right).$$ {#eq:GRPO}

PPO와 비교하여, GRPO의 표준 구현은 KL 발산을 손실에 포함한다는 점에 주목하라.
위와 같이, 이를 토큰별 계산으로 확장할 수 있다:

$$\begin{aligned}
J(\theta) = \frac{1}{G}\sum_{i=1}^G  \frac{1}{|a_i|} \sum_{t=1}^{|a_i|} \Bigg( &\min\!\left(\frac{\pi_\theta(a_{i,t}|s_{i})}{\pi_{\theta_{\text{old}}}(a_{i,t}|s_{i})}A_{i,t},\; \text{clip} \left( \frac{\pi_\theta(a_{i,t}|s_{i})}{\pi_{\theta_{\text{old}}}(a_{i,t}|s_{i})}, 1-\varepsilon, 1+\varepsilon \right) A_{i,t} \right) \\
&- \beta \mathcal{D}_{\text{KL}}\!\left(\pi_\theta(\cdot|s_{i})\|\pi_{\text{ref}}(\cdot|s_{i})\right) \Bigg)
\end{aligned}$$ {#eq:GRPO_token}


완성 인덱스 $i$에 대한 이점 계산:

$$A_i = \frac{r_i - \text{mean}({r_1, r_2, \cdots, r_G})}{\text{std}({r_1, r_2, \cdots, r_G})}.$$ {#eq:GRPO_ADV}

![GRPO 구조. 이점은 그룹 평균 및 표준 편차에 대해 정규화된다. KL 페널티는 보상을 조정하는 대신 손실에 직접 적용된다.](images/grpo_tikz.png){#fig:grpo-arch data-dark-src="images/grpo_tikz-dark.png"}

직관적으로, GRPO 업데이트는 배치 내의 단일 질문에 대한 여러 답변을 비교한다.
모델은 정답으로 표시된 답변처럼 더 많이, 그리고 다른 답변처럼 덜 행동하도록 학습한다.
이것은 이점(특정 행동이 주어진 상태에서 평균 대비 얼마나 더 나은지의 척도)을 계산하는 매우 단순한 방법이다.
PPO, REINFORCE, 그리고 보상 모델 평가(상대적 출력 보상)와 함께 수행되는 RLHF에 비해, GRPO는 이점이 전적으로 해당 프롬프트의 동료 완성들에 대한 완성의 상대적 가치에 관한 것이기 때문에 프롬프트당 훨씬 더 많은 샘플로 실행되는 경우가 많다.
여기서 현재 정책은 주어진 프롬프트에 대해 여러 응답을 생성하고, 그룹별 GRPO 이점 추정은 가치 있는 맥락을 얻는다.
PPO 및 기본 정책 그래디언트 알고리즘은 모든 완성의 보상을 정확하게 추정하도록 설계되었다 (실제로 일부 경우에는 더 많은 완성이 가치 추정을 거의 개선하지 못할 수도 있다).
GRPO 및 그 변형들은 주어진 프롬프트에 대한 여러 완성이 매우 자연스러운 현대 언어 모델 도구에 특히 잘 적합하다 (예를 들어 로봇 작업의 설정된 환경 상태에서의 여러 행동과 비교했을 때).

GRPO의 이점 계산은 편향에 트레이드오프가 있다.
표준 편차로 정규화하는 것은 배치에서 답변 정확도 변동이 낮은 질문에 보상을 준다.
거의 모두 정답이거나 모두 오답인 질문의 경우, 표준 편차가 낮아지고 이점이 높아진다.
Liu et al. 2025 [@liu2025understanding]는 이 편향을 고려하여 표준 편차 항 제거를 제안하지만, 이는 오답이 거의 없고 정답이 몇 개 있는 질문에 대한 가중치를 낮추는 대가를 치르게 된다. 이는 모델에게 가치 있는 학습 신호가 될 수 있다.
그러한 고분산 프롬프트들은 정확히 가장 어려운 경우일 수 있으며, 샘플링된 완성 중 일부만이 정답을 찾아 강한 훈련 신호를 제공한다.

@eq:GRPO_ADV 는 결과 감독(표준 보상 모델 또는 단일 검증 가능한 보상)으로 작업할 때 GRPO를 구현하는 방식이며, 과정 감독에는 다른 구현이 필요하다.
이 경우 GRPO는 다음 추론 단계에 대한 정규화된 보상의 합으로 이점을 계산한다.

마지막으로, GRPO의 이점 추정은 PPO 클리핑 없이 더 기본적인 정책 그래디언트 버전(예: REINFORCE)에도 적용될 수 있지만, 이것은 정식 형태가 아니다.
이 알고리즘들이 어떻게 서로 얽혀있는지의 예시로, GRPO의 변형인 Dr. GRPO (GRPO Done Right) [@liu2025understanding]에서의 이점 추정이 상수 스케일링 인수(일반적으로 이점을 정규화하는 구현 세부사항으로 인해 문제가 되지 않음)까지 RLOO 추정(다른 샘플의 평균 보상을 기준선으로 사용)과 동등함을 보일 수 있다.
Dr. GRPO는 @eq:GRPO_ADV 에서 표준 편차 정규화 항을 제거한다 — 이것은 또한 이점을 *증가*시키는 효과가 있는데, 이는 답변 점수의 분산이 있는 샘플에 대해 GRPO 학습률을 증가시키는 것과 동등하다.
이는 거의 모든 답변이 맞거나 틀린 경우처럼 보상 분산이 낮은 질문에 대한 편향을 해결하지만, 단 하나의 샘플만 정답을 맞추는 문제가 학습하기 중요할 수 있는 잠재적 비용을 수반한다.
크기 $G$의 그룹 내 완성 $i$에 대한 Dr. GRPO 이점은 다음과 같이 정의된다:

$$ \tilde{A}_i = r_i - \text{mean}({r_1, r_2, \cdots, r_G}) = r_i - \frac{1}{G}\sum_{j=1}^G r_j $$ {#eq:DrGRPO_ADV}

동일한 표기법에서, RLOO 이점 추정을 다시 상기하면:

$$ A_i^\text{RLOO} = r_i - \frac{1}{G-1}\sum_{j=1, i\neq j}^G r_j $$ {#eq:RLOO_ADV_AGAIN}

따라서 Dr. GRPO 이점 정의에 $\frac{G}{G-1}$을 곱하면 스케일링된 동등성을 확인할 수 있다:
$$
\begin{aligned}
\frac{G}{G-1} \tilde{A}_i &= \frac{G}{G-1} \left( r_i - \frac{1}{G}\sum_{j=1}^G r_j \right) \\
&= \frac{G}{G-1} r_i - \frac{1}{G-1} \sum_{j=1}^G r_j \\
&= \frac{G}{G-1} r_i - \frac{1}{G-1} \sum_{j=1, j\neq i}^G r_j - \frac{1}{G-1} r_i \\
&= r_i \left( \frac{G}{G-1} - \frac{1}{G-1} \right) - \frac{1}{G-1} \sum_{j=1, j\neq i}^G r_j \\
&= r_i - \frac{1}{G-1} \sum_{j=1, j\neq i}^G r_j \\
&= A_i^{\text{RLOO}}
\end{aligned}
$$ {#eq:RLOO_GRPO_EQUIV}

### 그룹 시퀀스 정책 최적화 (GSPO)

이전 정책에서 수집한 데이터 배치에 대해 여러 번의 그래디언트 스텝을 수행할 때, 데이터 수집 정책과 현재 최적화 중인 정책 사이의 분포 불일치를 보정하기 위해 중요도 샘플링이 필요합니다.
표준 중요도 샘플링 항등식은 한 분포에서 추출한 샘플을 사용하여 다른 분포에 대한 기댓값을 추정할 수 있게 해줍니다:

$$
\mathbb{E}_{p}[f(x)] = \mathbb{E}_{q}\left[f(x) \frac{p(x)}{q(x)}\right],
$$ {#eq:IS_identity}

여기서 $p$는 목표 분포, $q$는 샘플링 분포, $\frac{p(x)}{q(x)}$는 중요도 가중치입니다.
정책 그래디언트 방법에서 $p = \pi_\theta$는 최적화하고자 하는 현재 정책이고, $q = \pi_{\theta_{\text{old}}}$는 훈련 데이터를 생성한 정책입니다.
이를 통해 $\pi_{\theta_{\text{old}}}$ 하에서 수집된 샘플을 재가중치하여 $\pi_\theta$에 대한 그래디언트를 추정할 수 있으며, 롤아웃 배치당 여러 번의 그래디언트 스텝이 가능해집니다.

이러한 분포 불일치는 두 가지 일반적인 시나리오에서 발생합니다: (1) 단일 배치에서 여러 번의 그래디언트 스텝을 수행하는 경우로, 각 업데이트 후 $\pi_\theta$가 $\pi_{\theta_{\text{old}}}$로부터 멀어지는 경우, 그리고 (2) 추론 백엔드(예: vLLM)와 훈련 백엔드(예: FSDP)가 동기화 지연으로 인해 서로 다른 모델 가중치를 가질 수 있는 비동기 훈련 시스템의 경우(이 장 후반의 비동기성 절에서 다루며, 검증 가능한 보상을 위한 RL에 대한 관심과 함께 특히 부각되었으나, RLHF 설정에서도 사용됩니다).

PPO와 GRPO는 토큰 수준에서 중요도 샘플링을 적용하고 *대리 목적함수*를 클리핑하여 학습을 안정화합니다.
그러나 이 방식에는 미묘한 실패 모드가 있습니다: 어떤 토큰의 중요도 비율이 클리핑 범위 $[1-\varepsilon, 1+\varepsilon]$ 밖으로 벗어나면 해당 토큰은 그래디언트를 전혀 받지 못합니다.
모델이 처음에 낮은 확률을 부여하는 핵심 추론 단계와 같이, 드물지만 중요한 토큰들에 대해 이러한 "토큰 드롭"이 발생하면 모델이 해당 토큰을 더 안정적으로 생성하도록 학습하는 것을 방해할 수 있습니다.

그룹 시퀀스 정책 최적화(GSPO) [@zheng2025gspo]는 토큰 수준이 아닌 시퀀스 수준에서 중요도 비율을 계산함으로써 GRPO를 확장합니다.
이 알고리즘과, 나중에 다룰 정책 그래디언트 알고리즘의 중요도 샘플링 계산 방식을 수정하는 동류 알고리즘인 CISPO의 실용적 동기는, 토큰별 중요도 샘플링 비율이 수치적으로 불안정한 경우가 많다는 데 있습니다.
개념적 동기는 보상이 시퀀스 수준에서 부여될 때(대부분의 RLHF 및 RLVR 설정처럼), 중요도 샘플링 보정도 해당 단위와 일치해야 한다는 것입니다.

토큰 수준의 비율은 긴 시퀀스 및/또는 크고 희소한 모델(예: 현대의 전문가 혼합, MoE 모델)에서 불규칙하게 동작할 수 있습니다: 비율이 큰 단일 토큰이 정책 업데이트를 지배하거나, 하나의 응답 내에서 많은 토큰들이 독립적으로 클리핑되어 단일 응답에 걸친 학습 신호가 분산될 수 있습니다.
GSPO는 응답당 단일 중요도 가중치를 계산함으로써 이 문제를 해결합니다.

전체 응답의 확률은 자기회귀적으로 분해됨을 상기하십시오:

$$
\pi_\theta(a \mid s) = \prod_{t=1}^{|a|} \pi_\theta(a_t \mid s, a_{<t}).
$$ {#eq:response_factorization}

단순화를 위해 조건부 정책 $\pi_\theta(a_t \mid s, a_{<t})$를 $\pi_\theta(a_t \mid s)$로 줄여 쓰는 경우가 많으며, 이는 완성 내의 이전 행동(토큰)들을 암묵적으로 포함합니다.
GSPO는 기하 평균을 사용하여 길이 정규화된 시퀀스 수준의 중요도 비율을 정의합니다(긴 시퀀스의 수치적 문제를 피하기 위해):

$$
\rho_i(\theta) = \left( \frac{\pi_\theta(a_i \mid s)}{\pi_{\theta_{\text{old}}}(a_i \mid s)} \right)^{\frac{1}{|a_i|}} = \exp\left( \frac{1}{|a_i|} \sum_{t=1}^{|a_i|} \log \frac{\pi_\theta(a_{i,t} \mid s, a_{i,<t})}{\pi_{\theta_{\text{old}}}(a_{i,t} \mid s, a_{i,<t})} \right).
$$ {#eq:GSPO_ratio}

GSPO 목적함수는 GRPO를 따르되 이 시퀀스 수준의 비율을 사용합니다:

$$
J_{\text{GSPO}}(\theta) = \mathbb{E}_{s \sim \mathcal{D},\, \{a_i\}_{i=1}^G \sim \pi_{\theta_{\text{old}}}(\cdot \mid s)} \left[ \frac{1}{G} \sum_{i=1}^G \min\left( \rho_i(\theta) A_i,\, \text{clip}(\rho_i(\theta), 1-\varepsilon, 1+\varepsilon) A_i \right) \right].
$$ {#eq:GSPO_objective}

비율이 길이 정규화되어 있으므로, 클리핑 범위 $\varepsilon$은 토큰당 평균 스케일에서 작동하여 서로 다른 길이의 응답에 걸쳐 유효 제약이 비교 가능해집니다.
구현 측면에서, 시퀀스 수준 가중치 $\rho_i$는 응답 $a_i$의 모든 토큰에 균일하게 적용되어, 시퀀스 수준의 IS 보정을 유지하면서 그래디언트 계산을 단순화합니다.

이점 함수 계산은 GRPO(@eq:GRPO_ADV)와 동일하게 그룹 상대적 평균 및 표준 편차 정규화를 사용하며, GRPO의 다른 파생 연구에서와 같이 수정될 수 있습니다.
GSPO는 "시퀀스 수준 중요도 비율을 가진 GRPO"로 요약할 수 있습니다—IS 보정의 단위가 보상의 단위와 일치합니다.

### 클리핑된 중요도 샘플링 정책 최적화 (CISPO)

클리핑된 중요도 샘플링 정책 최적화(CISPO) [@minimax2025minimax_m1]는 다른 접근 방식을 취합니다: 대리 목적함수를 클리핑하는 대신, CISPO는 중요도 가중치 자체를 클리핑하면서 모든 토큰의 그래디언트를 보존합니다.
이 목적함수는 클리핑된 중요도 가중치에 정지 그래디언트(stop-gradient)를 사용하여, PPO 방식의 양방향 클리핑 대신 REINFORCE 방식의 공식으로 되돌아갑니다:

$$
J_{\text{CISPO}}(\theta) = \mathbb{E}_{s \sim \mathcal{D},\, \{a_i\}_{i=1}^K \sim \pi_{\theta_{\text{old}}}(\cdot \mid s)} \left[ \frac{1}{\sum_{i=1}^K |a_i|} \sum_{i=1}^K \sum_{t=1}^{|a_i|} \text{sg}\left( \hat{\rho}_{i,t}(\theta) \right) A_{i,t} \log \pi_\theta(a_{i,t} \mid s, a_{i,<t}) \right],
$$ {#eq:CISPO_objective}

여기서 $\text{sg}(\cdot)$는 정지 그래디언트를 의미하며(가중치는 사용되지만 역전파되지 않음), 클리핑된 중요도 비율은 다음과 같습니다:

$$
\hat{\rho}_{i,t}(\theta) = \text{clip}\left( \rho_{i,t}(\theta),\, 1 - \varepsilon_{\text{low}},\, 1 + \varepsilon_{\text{high}} \right), \quad \rho_{i,t}(\theta) = \frac{\pi_\theta(a_{i,t} \mid s, a_{i,<t})}{\pi_{\theta_{\text{old}}}(a_{i,t} \mid s, a_{i,<t})}.
$$ {#eq:CISPO_ratio}

PPO/GRPO와의 핵심 차이는 미묘하지만 중요합니다: 가중치를 클리핑하는 것(목적함수가 아닌)은 모든 토큰이 이점 함수에 비례하는 그래디언트 신호를 여전히 받는다는 것을 의미합니다—가중치는 단지 중요도 비율에 의해 해당 신호가 얼마나 증폭되거나 억제되는지를 제한할 뿐입니다.
이는 편향-분산 트레이드오프입니다: 가중치 클리핑은 편향을 유발하지만 분산을 제어하고, 결정적으로 토큰 그래디언트가 완전히 소실되는 것을 방지합니다.

CISPO와 GSPO 모두 수치적 문제로 알려진 대규모 MoE 모델에 RL을 적용하는 한계를 밀어붙이는 조직들에 의해 개발되었습니다.
이 논문들은 토큰별 중요도 샘플링 비율이 불안정하고 그래디언트에 상당한 분산을 더하여 학습을 저해할 수 있음을 강조합니다.
이는 이러한 알고리즘들이 대규모 모델에서 특히 효과적일 수 있으나, 소규모의 학술 실험에서는 덜 연구되고 덜 유익할 수 있음을 의미합니다.

CISPO는 또한 비대칭 클리핑 범위($\varepsilon_{\text{low}} \neq \varepsilon_{\text{high}}$)를 허용하는데, 이는 이 장 후반에서 다룰 DAPO의 "높게 클리핑" 수정과 유사하며, 모델이 확률을 높이고자 하는 토큰에 대해 더 큰 업데이트를 허용함으로써 탐색을 장려할 수 있습니다.
관련 연구로는 테이퍼드 오프-정책 REINFORCE(TOPR) [@leroux2025topr]가 있으며, 이는 목적함수 내부에서 클리핑하는 PPO/GRPO와 달리 IS 가중치를 직접 클리핑하고(CISPO처럼), 시퀀스 수준에서 작동하며(GSPO처럼), 보상 부호에 따른 비대칭 클리핑을 사용합니다—양의 보상에 대해서는 IS 보정을 적용하지 않고, 음의 보상에 대해서는 비율을 $[0, 1]$로 클리핑합니다—안정적인 오프-정책 학습을 가능하게 합니다.


## 구현

이러한 알고리즘들이 개발된 원래의 심층 RL 문헌과 비교하여, 언어 모델이나 다른 대형 AI 모델을 최적화하기 위한 RL 구현에는 많은 세부적인 구현 사항이 필요합니다.
이 절에서는 인기 있는 알고리즘 구현을 차별화하는 몇 가지 핵심 요소를 강조합니다.

이 훈련 과정에는 다른 많은 세부 사항들도 포함됩니다.
예를 들어, 언어 모델로 RLHF를 수행할 때 보상 모델이 평가할 텍스트를 생성하는 것이 중요한 단계입니다.
일반적인 상황에서 모델은 생성을 마쳤음을 나타내는 시퀀스 종료(EOS) 토큰을 생성해야 하지만, 인프라를 효율적으로 활용하기 위해 생성 길이에 하드 상한선을 두는 것이 일반적인 관행입니다.
RLHF의 실패 모드 중 하나는 모델의 응답이 정기적으로 잘려서 보상 모델의 점수가 분포 밖으로 벗어나 예측 불가능한 점수를 받는 것입니다.
이에 대한 해결책은 보상 모델 채점을 `eos_token`에서만 실행하고, 그 외의 경우 너무 길게 생성하는 모델에 패널티를 부여하는 것입니다.

RLHF를 위한 인기 오픈 소스 도구들은 알고리즘 전반에 걸쳐 구현 세부 사항에서 큰 편차를 보입니다([@ivison2024unpacking]의 표 10 참조).
여기서 다루지 않는 몇 가지 결정 사항은 다음과 같습니다:

- **가치 네트워크 초기화**: PPO 및 유사 알고리즘에서 사용하는 내부 학습 가치 네트워크는 동일한 아키텍처의 다른 모델이나 무작위로 선택된 가중치로 시작할 수 있습니다. 이는 성능에 큰 영향을 미칠 수 있습니다. InstructGPT [@ouyang2022training]에서 확립된 표준(및 RLVR 작업을 위해 Tülu 3 [@lambert2024t]에서 재사용)은 RLHF 중에 사용된 보상 모델로 가치 네트워크를 초기화하는 것입니다. 다른 방법으로는 무작위로 초기화된 가치 헤드가 추가된 RLHF 훈련 이전 체크포인트(보통 SFT 모델)를 사용하거나, 완전히 재초기화된 언어 모델을 사용하는 경우도 있습니다(RLHF가 수렴하는 데 더 오래 걸리므로 덜 일반적이지만 가능합니다).
- **보상 정규화, 보상 화이트닝, 및/또는 이점 함수 화이트닝**: 정규화는 RM(또는 환경)의 모든 값을 0과 1 사이로 제한하여 학습 안정성에 도움을 줍니다. [화이트닝](https://en.wikipedia.org/wiki/Whitening_transformation)은 보상이나 이점 함수 추정치를 평균 0, 분산 1로 변환하여 안정성을 더욱 강화합니다.
- **다양한 KL 추정기**: 복잡한 언어 모델에서 모델 간 KL 발산을 정확하게 계산하는 것은 복잡할 수 있으므로, 정확한 계산을 대체하기 위한 여러 근사법이 사용됩니다 [@schulman2016klapprox].
- **KL 컨트롤러**: PPO 및 관련 알고리즘의 초기 구현에는 특정 KL을 목표로 하고 최근 측정값을 기반으로 패널티를 변경하는 동적 컨트롤러가 있었습니다. 대부분의 현대 RLHF 구현은 정적 KL 패널티를 사용하지만, 이는 구현에 따라 다를 수 있습니다.

RLHF 구현 세부 사항에 대한 더 많은 정보는 [@huang2024n]을 참조하십시오.
알고리즘에 대한 추가 정보는 [@weng2018PG]를 참조하십시오.

### 정책 그래디언트 기초

PPO 및 GRPO와 같은 고급 알고리즘을 준비하기 위해 이점 함수를 사용하는 정책 그래디언트의 간단한 구현은 다음과 같습니다:
```python
pg_loss = -advantages * ratio
```
여기서 ratio는 참조 모델의 확률 대비 새 정책 모델 확률의 (토큰별) 확률 비율입니다(보통 로그 확률 차이로 계산됩니다).

이 수식을 이해하기 위해 업데이트 배치 내에서 발생할 수 있는 다양한 경우를 살펴보는 것이 좋습니다.
손실이 모델이 작업을 잘 수행할수록 *감소*해야 함을 기억하십시오.

경우 1: 양의 이점 함수 값, 즉 행동이 상태의 기댓값보다 더 좋았습니다. 이를 강화하고 싶습니다. 이 경우 모델은 음의 부호로 인해 이것을 더 발생하기 쉽게 만듭니다. 이를 위해 로그 비율을 증가시킵니다. 양의 로그 비율, 즉 토큰들의 로그 확률 합은 모델이 해당 토큰들을 생성할 가능성이 더 높다는 것을 의미합니다.

경우 2: 음의 이점 함수 값, 즉 행동이 상태의 기댓값보다 더 나빴습니다. 이는 매우 유사하게 따릅니다. 여기서 새 모델이 더 높은 확률을 부여했다면 손실이 양수가 될 것이므로, 모델은 정책 매개변수가 이 완성을 덜 발생하기 쉽게 만들려 할 것입니다.

경우 3: 이점 함수 값이 0, 즉 업데이트가 필요하지 않습니다. 손실이 0이므로 정책 모델을 변경하지 않습니다.

### 손실 집계 트레이드오프

언어 모델로 정책 그래디언트 알고리즘을 구현할 때의 핵심 질문은: 토큰별 손실을 최종 스칼라 손실로 어떻게 집계하느냐입니다.
배치 크기 $B$에서 토큰 $t$의 샘플 $i$에 대한 토큰별 손실 $\ell_{i,t}$, 완성 길이 $|a_i|$가 주어질 때, 세 가지 주요 전략이 있습니다:

**전략 1: 시퀀스별 정규화** (표준 GRPO; 일부 PPO 구현에서도 사용)

$$L = \frac{1}{B} \sum_{i=1}^{B} \frac{1}{|a_i|} \sum_{t=1}^{|a_i|} \ell_{i,t}$$ {#eq:loss_per_sequence}

각 시퀀스는 길이에 관계없이 배치 손실에 동등하게 기여합니다. 코드로는:

```python
# Strategy 1: Per-sequence normalization
sequence_loss = ((per_token_loss * completion_mask).sum(dim=1) / \
             completion_mask.sum(dim=1)).mean()
```

**전략 2: 토큰별 정규화** (DAPO [@yu2025dapo])

$$L = \frac{\sum_{i=1}^{B} \sum_{t=1}^{|a_i|} \ell_{i,t}}{\sum_{i=1}^{B} |a_i|}$$ {#eq:loss_per_token}

각 토큰이 동등하게 기여하며, 더 긴 시퀀스는 그래디언트에 비례적으로 더 많은 영향을 미칩니다. 코드로는:

```python
# Strategy 2: Per-token normalization
token_loss = ((per_token_loss * completion_mask).sum() / \
            completion_mask.sum())
```

**전략 3: 고정 길이 정규화** (Dr. GRPO [@liu2025understanding])

$$L = \frac{1}{B} \sum_{i=1}^{B} \frac{1}{L_{\max}} \sum_{t=1}^{|a_i|} \ell_{i,t}$$ {#eq:loss_fixed_length}

최대 시퀀스 길이 $L_{\max}$로 정규화하여, 더 긴 시퀀스가 더 많은 활성 토큰을 포함하기 때문에 여전히 더 많은 총 그래디언트를 기여하면서도 시퀀스에 걸쳐 토큰별 스케일을 균등화합니다. 코드로는:

```python
# Strategy 3: Fixed-length normalization
fixed_len_loss = ((per_token_loss * completion_mask).sum(dim=1) / \
            L_max).mean()
```

여기서 $L_{\max}$는 일반적으로 전체 훈련 과정에서 전역 상수로, 최대 생성 토큰 수를 지정합니다.

위 코드에서 `completion_mask`는 1과 0으로 이루어진 행렬로, 프롬프트 토큰들은 마스킹됩니다(0으로 설정). 모델이 프롬프트 토큰 예측에서 학습하는 것을 원하지 않기 때문입니다.

#### 왜 이것이 중요한가?

직관적으로 우리는 개별 토큰이 아닌 *결과*에 관심이 있으므로 시퀀스별 정규화(전략 1)가 가장 좋아 보입니다.
그러나 이는 시퀀스 길이에 따른 미묘한 편향을 도입하여, 편향의 방향에 따라 모델이 과잉 추론하거나 자연스럽게 더 많은 토큰이 필요한 전략의 가중치를 낮추는 원인이 될 수 있습니다.
토큰별 손실이 다른 두 시퀀스를 고려해 보십시오:

```python
seq_1_losses = [1, 1, 1, 1, 10]  # 5 tokens, mean = 2.8
seq_2_losses = [1, 1, 1, 1, 1, 1, 1, 1, 1, 10]  # 10 tokens, mean = 1.9
```

**전략 1**(시퀀스별): 배치 손실은 $(2.8 + 1.9)/2 = 2.35$이며, 결정적으로 짧은 시퀀스의 각 토큰이 긴 시퀀스의 토큰보다 더 큰 그래디언트를 받습니다.

**전략 2**(토큰별): 배치 손실은 $(14 + 19)/15 = 2.2$이며, 모든 토큰이 동일한 그래디언트 크기를 받습니다.

**전략 3**(고정 길이, $L_{\max}=10$ 사용): 짧은 시퀀스는 $1.4$, 긴 시퀀스는 $1.9$를 기여하여, 시퀀스 길이로 가중치를 부여하면서도 토큰별 그래디언트의 균형을 맞춥니다.

이러한 전략들이 그래디언트에 어떤 영향을 미치는지 더 완전한 예시는 아래 스크립트를 참조하십시오.

```python
from typing import Optional
import torch

def masked_mean(values: torch.Tensor, mask: torch.Tensor, axis: Optional[int] = None) -> torch.Tensor:
    """Compute mean of tensor with masked values."""
    if axis is not None:
        return (values * mask).sum(axis=axis) / mask.sum(axis=axis)
    else:
        return (values * mask).sum() / mask.sum()

def masked_sum(
        values: torch.Tensor,
        mask: torch.Tensor,
        axis: Optional[int] = None,
        constant_normalizer: float = 1.0,
    ) -> torch.Tensor:
    """Compute sum of tensor with masked values. Use a constant to normalize."""
    if axis is not None:
        return (values * mask).sum(axis=axis) / constant_normalizer
    else:
        return (values * mask).sum() / constant_normalizer

ratio = torch.tensor([
    [1., 1, 1, 1, 1, 1, 1,],
    [1, 1, 1, 1, 1, 1, 1,],
], requires_grad=True)


advs = torch.tensor([
    [2, 2, 2, 2, 2, 2, 2,],
    [2, 2, 2, 2, 2, 2, 2,],
])

masks = torch.tensor([
    # generation 1: 4 tokens
    [1, 1, 1, 1, 0, 0, 0,],
    # generation 2: 7 tokens
    [1, 1, 1, 1, 1, 1, 1,],
])

max_gen_len = 7

masked_mean_result = masked_mean(ratio * advs, masks, axis=1)
masked_mean_token_level = masked_mean(ratio, masks, axis=None)
masked_sum_result = masked_sum(ratio * advs, masks, axis=1, constant_normalizer=max_gen_len)

print("masked_mean", masked_mean_result)
print("masked_sum", masked_sum_result)
print("masked_mean_token_level", masked_mean_token_level)

# masked_mean tensor([2., 2.], grad_fn=<DivBackward0>)
# masked_sum tensor([1.1429, 2.0000], grad_fn=<DivBackward0>)
# masked_mean_token_level tensor(1., grad_fn=<DivBackward0>)

masked_mean_result.mean().backward()
print("ratio.grad", ratio.grad)
ratio.grad.zero_()
# ratio.grad tensor([[0.2500, 0.2500, 0.2500, 0.2500, 0.0000, 0.0000, 0.0000],
# [0.1429, 0.1429, 0.1429, 0.1429, 0.1429, 0.1429, 0.1429]])

masked_sum_result.mean().backward()
print("ratio.grad", ratio.grad)
ratio.grad.zero_()
# ratio.grad tensor([[0.1429, 0.1429, 0.1429, 0.1429, 0.0000, 0.0000, 0.0000],
# [0.1429, 0.1429, 0.1429, 0.1429, 0.1429, 0.1429, 0.1429]])

masked_mean_token_level.mean().backward()
print("ratio.grad", ratio.grad)
# ratio.grad tensor([[0.0909, 0.0909, 0.0909, 0.0909, 0.0000, 0.0000, 0.0000],
# [0.0909, 0.0909, 0.0909, 0.0909, 0.0909, 0.0909, 0.0909]])
```

출력 결과는 전략 1(`masked_mean`)을 사용했을 때 짧은 시퀀스의 토큰별 그래디언트(0.25)가 긴 시퀀스(0.14)보다 더 크다는 것을 보여줍니다.
전략 2와 3은 시퀀스에 걸쳐 토큰별 그래디언트를 균등화합니다.
그래디언트 누적이 사용될 경우 이러한 결과는 상당히 달라질 수 있음에 주의하십시오—역전파 스텝을 수행하기 전에 여러 미니배치에 걸쳐 그래디언트가 합산되는 경우, 더 짧은 시퀀스와 더 긴 시퀀스 사이의 균형이 뒤집힐 수 있습니다.

실제로 최선의 전략은 특정 훈련 설정에 따라 다릅니다.
RLHF에서는 수치적 안정성이 가장 좋거나 손실의 분산이 가장 적은 방법이 선호되는 경우가 많습니다.

#### 관련: MDP 대 밴딧 프레임

손실 집계의 선택은 RL 문제를 어떻게 프레임하는지에 대한 더 깊은 구분과 연결됩니다.
**MDP(토큰 수준)** 관점은 각 토큰 $a_t$를 상태 $s_t$가 현재까지의 프리픽스인 행동으로 취급합니다.
실제로 이는 학습된 가치 함수 $V(s_t)$(예: GAE [@schulman2015high])로 토큰 수준 이점 함수를 계산하고 토큰별 KL 패널티를 적용할 때 사용되는 프레임입니다.
학습된 가치 네트워크를 사용하는 PPO가 대표적인 예입니다 [@schulman2017proximal].

반면, **밴딧(시퀀스 수준)** 관점은 전체 완성을 하나의 스칼라 보상 $R$을 가진 단일 행동으로 취급합니다.
코드에서 이는 시퀀스 수준의 이점 함수 $A_{\text{seq}}$를 계산하고 모든 토큰에 브로드캐스팅하는 것을 의미합니다.
RLOO와 GRPO 방식의 이점 함수는 종종 이 밴딧 방식 설정에서 사용됩니다 [@kool2019buy] [@ahmadian2024back] [@shao2024deepseekmath].
DPO와 A-LoL 같은 직접 정렬 방법도 시퀀스 수준 목적함수를 정의하지만, 정책 그래디언트 추정기는 아닙니다 [@baheti2023leftover].

많은 GRPO 구현이 밴딧 방식의 이점 함수를 사용하면서 손실에 별도의 토큰별 KL 항을 추가하고, 많은 PPO/RLOO 구현은 이점 함수를 계산하기 전에 KL을 보상에 포함시킨다는 점에 주목하십시오; 두 관행 모두 실제로 존재합니다.

두 접근 방식을 비교하는 예시는 아래와 같습니다:

```python
# === Bandit-style (sequence-level) ===
# One scalar reward per sequence; advantage broadcast to all tokens
reward = torch.tensor([3.0, 1.0])       # (B,) e.g., reward model scores
baseline = reward.mean()                 # simple baseline (RLOO uses leave-one-out)
advantage_seq = reward - baseline        # (B,)
advantages = advantage_seq[:, None].expand(-1, seq_len)  # (B, L)
# tensor([[ 1.,  1.,  1.,  1.],    <- same advantage for all tokens
#         [-1., -1., -1., -1.]])

# === MDP-style (token-level) ===
# Per-token rewards + learned V(s_t); each token gets its own advantage
# (could also use per-token KL shaping, format rewards, or other token-level signals)
advantages = gae(per_token_rewards, values, done_mask, gamma=1.0, lam=0.95)
# tensor([[ 0.2,  0.5,  0.8,  1.5],    <- varies by position
#         [-0.3, -0.5, -0.8, -1.4]])
```

이 프레임 구분은 또한 할인 인수 $\gamma$가 거의 모든 RLHF 구현에서 1.0으로 설정되는 이유를 설명합니다.
표준 RL에서 할인($\gamma < 1$)은 필수적입니다: 다단계 에피소드에 걸쳐 단기 및 장기 보상 간의 최적화 균형을 맞추는 것이 에이전트가 시간에 걸쳐 효과적인 행동을 학습하는 데 중요합니다.
그러나 RLHF 설정에서는 토큰 수준의 MDP 관점을 사용하더라도 최적화의 귀납적 편향은 집합적 완성의 품질에 있습니다—보상 신호는 개별 토큰이 아닌 전체 응답을 평가합니다.
앞의 토큰들을 할인하는 것은 원칙적 정당성 없이 임의로 그 기여를 낮추는 것이 됩니다.
에이전트형 RL 설정이 성숙해짐에 따라—모델이 도구 호출, 코드 실행, 웹 브라우징 같은 실제 다단계 행동을 취하는—할인이 다시 관련성을 가질 수 있습니다. 이러한 경우 장기적 결과가 다른 진정으로 구별되는 순차적 결정을 포함하기 때문입니다.

### 비동기 RL 시스템

정책 그래디언트 알고리즘의 기본 구현은 **온-정책** 실행이라 불리는 것으로, 에이전트(언어 모델)가 취한 행동(생성)이 모델을 업데이트하기 전에 평가됩니다.
정책 그래디언트의 이론적 유도는 모든 행동이 최신 시험/롤아웃 결과와 항상 최신 상태인 모델과 정확히 온-정책임을 전제합니다.
실제로 정확한 온-정책 실행을 유지하는 것은 훈련을 상당히 느리게 합니다 [@noukhovitch2024asynchronous]—그리고 완벽한 동기화는 어차피 기술적으로 불가능합니다.
따라서 언어 모델을 활용한 최근의 모든 실증적 결과는 이론적 증명의 범위를 약간 벗어나는 경향이 있습니다.
실제로 일어나는 일은 실제로 작동하는 것에 맞게 알고리즘과 시스템을 설계하는 것입니다.

![Noukhovitch et al. 2024를 따른 동기식 또는 비동기식 RL 훈련의 생성-업데이트 단계 비교.](images/async_v_synch_rl.png){#fig:async}

일반적으로 사용되는 해결책은 @fig:async의 하단에 표시된 것처럼 별도의 GPU 노드에서 추론과 훈련을 지속적으로 실행하도록 설계된 소프트웨어를 사용하는 것입니다.
언어 모델용 인기 오픈 소스 RL 도구에서 일반적인 관행은 Ray와 같은 분산 프로세스 관리 라이브러리를 사용하여 정책 그래디언트 학습 루프와 vLLM과 같은 효율적인 추론 엔진을 사용하는 추론 루프 사이에 정보를 전달하는 것입니다.
이러한 설정에서 RL 스텝을 수행하는 데 전용된 GPU는 "학습자(learner)"라 불리고, 언어 모델에서 샘플링하는 데 전용된 GPU는 "행위자(actor)"라 불립니다.
훈련을 더 비동기적으로 만들 때 직면하는 주요 과제는 훈련 안정성 유지와 학습 신호 보존입니다.

![두 개의 큐가 관리되어 학습자 및 행위자 GPU에 데이터를 전달하는 예시 분산 RL 시스템으로, Ray와 같은 분산 컴퓨팅 라이브러리로 동기화될 수 있습니다. Olmo Team 2025, 라이선스 CC-BY.](images/distributed-rl.png){#fig:async_system}

이러한 시스템은 거의 온-정책 데이터가 안정적인 학습에 충분하다는 가정 하에 설계되고 구현됩니다.
여기서 생성 및 업데이트 단계는 @fig:async_system에서 학습자에서 행위자로 모델 가중치를 전달하는 훈련 시스템의 어느 부분에서도 유휴 연산이 발생하지 않도록 쉽게 동기화될 수 있습니다.
추론 모델의 경우, 답당 10K에서 100K+ 토큰이 필요한 문제의 극도로 긴 추론 특성이 롤아웃 생성을 훨씬 더 강한 병목으로 만듭니다.
더 동기식 RL 인프라에서 추론 모델을 훈련할 때의 일반적인 문제는 배치 내 한 프롬프트에 대한 답변이 생성하는 데 상당히 더 많은 시간이 걸릴 수 있다는 것입니다(더 많은 토큰이나 도구 호출을 통해), 이로 인해 할당된 컴퓨팅의 대부분이 완료될 때까지 유휴 상태가 됩니다.
이 길이 불일치 문제에 대한 두 번째 해결책인 시퀀스 수준 패킹(sequence-level packing)은 배치 내 더 짧은 샘플들을 영리한 마스킹으로 쌓아 모델에서의 지속적인 롤아웃을 가능하게 하고 배치 내 샘플에 걸쳐 길이 정규화를 더 잘 분배합니다.
분산 RL 인프라의 전체 복잡성은 이 책의 범위를 벗어납니다. 이는 훈련을 느리게 하거나 불안정성을 유발하는 다른 많은 미묘한 문제들을 야기할 수 있기 때문입니다.

이러한 추론 모델의 등장에 따라, 훈련 및 추론 루프를 완전히 오프-정책으로 만드는 것에 대한 관심이 더욱 높아졌습니다. 여기서 정책 그래디언트 업데이트를 위한 훈련 배치는 여러 인스턴스에서 가장 최근에 완료된 롤아웃으로 채워집니다 [@wu2025llamarl] [@fu2025areal].
완전히 비동기적인 훈련은 학습자 노드(정책 그래디언트 스텝을 수행하는)와 행위자(문제를 해결하려는) 사이의 가중치 동기화 간격 증가 옵션으로 인해 여러 데이터센터에 걸쳐 RL 훈련 실행을 더 쉽게 확장할 수 있게 할 것입니다 [@primeintellectteam2025intellect2reasoningmodeltrained].

관련 방법들은 완전히 오프-정책 정책 그래디언트 알고리즘을 탐색하고 있습니다 [@leroux2025topr].

### 절단된 중요도 샘플링 (TIS)

절단된 중요도 샘플링(TIS)은 언어 모델을 사용하는 현대의 비동기 RL 프레임워크에서 훈련을 안정화하는 데 사용되는 중요한 도구입니다.
중요도 샘플링은 한 분포에서 추출된 샘플을 재가중하여 다른 분포에 대한 기댓값을 추정하는 보정 방법입니다(@eq:IS_identity에서 소개됨).
절단된 중요도 샘플링 [@ionides2008truncated]은 어떤 상수 $C$에 대해 $\min(\rho, C)$로 이 가중치에 상한선을 두어, 정책 그래디언트의 유계된 분산을 위해 약간의 편향을 트레이드합니다.

이는 정책 그래디언트에 적용되는 중요도 샘플링 보정이지만, 비율을 1 근처로 제한하는 PPO와 CISPO의 양방향 클리핑과 달리, TIS는 비율이 1 아래로 자유롭게 떨어질 수 있는 단방향 상한선을 사용하되 극단적인 과대 가중치를 방지하기 위해 $C$에서 상한을 둡니다.
PPO, GRPO, CISPO(및 관련 알고리즘) 모두에서, 비율 $\rho_t^{\text{policy}} = \pi_\theta(a_t \mid s) / \pi_{\theta_{\text{old}}}(a_t \mid s)$는 하나의 RL 배치 내 여러 그래디언트 스텝에 걸친 정책 드리프트를 보정합니다.
이전 소절의 비동기성 아이디어를 중심으로 한 실제 RL 프레임워크로 전환할 때, 더 큰 수치적 차이 원인이 있을 수 있습니다(이 역시 중요도 샘플링의 수치 보정이 필요합니다).
샘플러와 학습자가 동일한 매개변수 $\theta$를 공유하더라도, 추론 엔진(예: vLLM)과 훈련 프레임워크(예: FSDP)가 서로 다른 커널, 정밀도, 병렬화 전략을 사용하기 때문에 유효 토큰 분포가 다를 수 있습니다 [@yao2025offpolicy].
따라서 두 시스템에서 평가된 동일한 정책 $\pi_\theta^{\text{sampler}}$와 $\pi_\theta^{\text{learner}}$를 구분하고, 대응하는 비율과 절단된 형태를 정의하는 것이 유용합니다:

$$
\rho_t^{\text{learner}} = \frac{\pi_\theta^{\text{learner}}(a_t \mid s, a_{<t})}{\pi_\theta^{\text{sampler}}(a_t \mid s, a_{<t})}, \qquad \tilde{\rho}_t^{\text{learner}} = \min(\rho_t^{\text{learner}},\; C).
$$ {#eq:tis_backend}

이 두 보정은 상호 보완적이지만, 서로 다른 이유로 정책 그래디언트 구현에 포함됩니다—하나는 RL 배치의 훈련 내 정책 드리프트를 보상하고, 다른 하나는 구현으로 인한 발산을 보상합니다—그리고 동시에 적용될 수 있습니다.
어떻게 결합되는지는 알고리즘에 따라 다릅니다:

**TIS를 사용하는 REINFORCE**(단일 그래디언트 스텝): 정책 드리프트가 없으므로($\pi_\theta = \pi_{\theta_\text{old}}$) 유일한 불일치는 학습자와 샘플러 사이에 있습니다.
여기서 $\pi_{\theta_\text{old}} = \pi_\text{gen}$이며, TIS가 직접 학습자-샘플러 간격을 보정합니다:

$$
\nabla_\theta J \approx \mathbb{E}_{a \sim \pi_\theta^{\text{sampler}}} \left[ \tilde{\rho}_t^{\text{learner}} \cdot A_t \cdot \nabla_\theta \log \pi_\theta^{\text{learner}}(a_t \mid s, a_{<t}) \right].
$$ {#eq:reinforce_tis}

**TIS를 사용하는 PPO/GRPO**(여러 그래디언트 스텝): 이제 두 비율이 모두 활성화됩니다.
신중한 구현에서, 정책 비율의 "이전 로그 확률"은 학습자에서 재계산되므로(GSPO 논문에서 이를 논의), 정책 비율 $\rho_t^{\text{policy}} = \pi_\theta^{\text{learner}} / \pi_{\theta_\text{old}}^{\text{learner}}$는 순수한 정책 드리프트를 포착하고, $\tilde{\rho}_t^{\text{learner}} = \min(\pi_{\theta_\text{old}}^{\text{learner}} / \pi_{\theta_\text{old}}^{\text{sampler}},\; C)$는 생성 체크포인트에서 백엔드 불일치를 별도로 보정합니다:

$$
J_{\text{PPO+TIS}}(\theta) = \mathbb{E}\left[ \min\!\left( \rho_t^{\text{policy}}\, A_t,\; \text{clip}\!\left(\rho_t^{\text{policy}}, 1-\varepsilon, 1+\varepsilon\right) A_t \right) \cdot \tilde{\rho}_t^{\text{learner}} \right].
$$ {#eq:ppo_tis}

여기서 $\pi_{\theta_\text{old}} \neq \pi_\text{gen}$입니다: 이전 로그 확률은 샘플러가 아닌 학습자에서 옵니다.
프레임워크가 이 재계산을 건너뛰고 샘플러 로그 확률을 $\pi_{\theta_\text{old}}$로 직접 사용하면, 정책 비율이 이미 백엔드 불일치를 포착하여 별도의 TIS 보정이 필요하지 않지만—그러나 클리핑이 어떠한 그래디언트 스텝 이전에도 이미 1.0에서 벗어난 더 노이즈가 많은 비율에서 작동하게 됩니다.
이것이 Yao et al. [-@yao2025offpolicy]의 "여러분의 프레임워크는 몰래 오프-정책 RL을 제공합니다" 관찰입니다.

실제로, LLM RL 시스템은 TIS를 정책 그래디언트 손실에 대한 토큰별 보정 가중치로 적용합니다:

```python
# Shape: (B*G, L)
C = 2.0  # TIS cap

logratio = learner_logprobs - sampler_logprobs
logratio = logratio.clamp(-10.0, 10.0)              # numerical safety
tis_weight = torch.exp(logratio).clamp(max=C)        # one-sided truncation

# Use as a fixed correction weight on the per-token PG loss
per_token_pg_loss = per_token_pg_loss * tis_weight.detach()
```

$[-10, 10]$ 클램프는 지수 계산 전 수치 안정성을 위한 것입니다; 실제 절단된 중요도 샘플링 스텝은 $C$에서의 단방향 상한선입니다.
실제로, 이러한 로그 확률 주변의 처리—생성 시 샘플러 로그 확률 저장, 이전 체크포인트에서 학습자 로그 확률 재계산, 그래디언트 스텝 중 현재 로그 확률 추적—는 분산 RL 프레임워크의 스캐폴딩에서 상당한 부분을 차지합니다.
GSPO와 달리, 이 보정은 시퀀스 수준의 보상 단위가 아닌 토큰 수준의 수치 불일치를 다루기 때문에 토큰 수준입니다.
학습자-샘플러 비율에 대한 TIS는 주요 오픈 소스 RL 프레임워크들(VeRL, TRL, OpenRLHF, SkyRL, OAT, 그리고 $C = 2$를 사용하는 Open Instruct)에 채택되었으며, 답당 수천 개의 생성 토큰에 걸쳐 작은 토큰별 차이가 누적되는 긴 추론 트레이스(7장)에서 점점 더 중요해집니다.


### 예시: 근위 정책 최적화

PPO의 구현은 매우 많습니다.
핵심 *손실* 계산은 아래에 나와 있습니다.
안정적인 성능을 위해서는 *가치* 계산도 중요한데, 여기에는 여러 가지 옵션이 존재합니다(*가치 모델* 손실에 대한 다양한 옵션 포함).

참조 정책(또는 이전 로그 확률)은 생성이 샘플링된 시점의 것이며, 반드시 참조 정책과 같은 것은 아닙니다.
참조 정책은 KL 거리 제약/패널티에만 사용됩니다.

```python
# B: Batch Size, L: Sequence Length, G: Num of Generations
# Apply KL penalty to rewards
rewards = rewards - self.beta * per_token_kl  # Shape: (B*G, L)

# Get value predictions
values = value_net(completions)  # Shape: (B*G, L)

# Compute returns via backward pass (gamma typically 1.0 for LM RLHF)
# Mask rewards to avoid padding tokens (which may have KL penalties) leaking into returns
returns = torch.zeros_like(rewards)
running = torch.zeros(rewards.shape[0], device=rewards.device, dtype=rewards.dtype)
for t in reversed(range(rewards.shape[1])):
    # Zero out padding: only accumulate rewards/returns for valid completion tokens
    running = (rewards[:, t] + self.gamma * running) * completion_mask[:, t]
    returns[:, t] = running

# Compute advantages: A_t = G_t - V(s_t)
advantages = returns - values.detach()  # Shape: (B*G, L)
# Note: We detach the value network here to not update the parameters of
# the value function when computing the policy-gradient loss

# Normalize advantages (optional but stable)
advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

# Compute probability ratio between new and old policies
ratio = torch.exp(new_per_token_logps - per_token_logps)  # Shape: (B*G, L)

# PPO clipping objective
eps = self.cliprange  # e.g. 0.2
pg_losses1 = -advantages * ratio  # Shape: (B*G, L)
pg_losses2 = -advantages * torch.clamp(ratio, 1.0 - eps, 1.0 + eps)  # Shape: (B*G, L)
pg_loss_max = torch.max(pg_losses1, pg_losses2)  # Shape: (B*G, L)

# Value function loss: predict returns
vf_loss = 0.5 * ((returns - values) ** 2)  # Shape: (B*G, L)

# Combine policy and value losses
per_token_loss = pg_loss_max + self.vf_coef * vf_loss  # Shape: (B*G, L)

# Apply completion mask and compute final loss
loss = ((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
 # Scalar

# Compute metrics for logging
with torch.no_grad():
    # Compute clipping fraction
    clip_frac = ((pg_losses2 > pg_losses1).float() * completion_mask).sum() / completion_mask.sum()

    # Compute approximate KL
    approx_kl = (0.5 * ((new_per_token_logps - per_token_logps)**2) * completion_mask).sum() / completion_mask.sum()

    # Compute value loss for logging
    value_loss = vf_loss.mean()
```

PPO에서 핵심적으로 이해해야 할 부분은 정책 그래디언트 손실이 어떻게 업데이트되는가입니다.
다음 세 줄에 집중하세요:
```python
pg_losses1 = -advantages * ratio  # Shape: (B*G, L)
pg_losses2 = -advantages * torch.clamp(ratio, 1.0 - eps, 1.0 + eps)  # Shape: (B*G, L)
pg_loss_max = torch.max(pg_losses1, pg_losses2)  # Shape: (B*G, L)
```
`pg_losses1`은 기본적인 이점 가중 정책 그래디언트 손실입니다. `pg_losses2`는 동일한 수식을 적용하되, 확률 비율을 $[1-\varepsilon, 1+\varepsilon]$ 범위로 클램핑하여 단일 업데이트에서 정책이 변할 수 있는 정도를 제한합니다.

핵심 아이디어는 두 손실 값에 `torch.max`를 취하는 것입니다. *음수* 손실을 최소화하고 있으므로(이점 앞의 음수 부호를 기억하세요), 최대값을 선택하면 더 비관적인 그래디언트, 즉 더 작은 정책 업데이트를 만들어내는 쪽이 선택됩니다. 이점이 양수인 경우(좋은 행동), 클리핑은 정책이 해당 행동의 확률을 너무 공격적으로 높이지 못하게 막습니다. 이점이 음수인 경우(나쁜 행동), 클리핑은 반대 방향으로의 과잉 수정을 방지합니다.

로그 확률 비율을 클램핑함으로써, PPO는 정책이 훈련 데이터를 생성한 버전에서 얼마나 벗어날 수 있는지를 제한하여, 명시적인 신뢰 영역 계산 없이도 학습을 안정화합니다.

위의 코드는 또한 PPO가 정책과 함께 가치 함수를 학습하는 것을 보여주는데, 이는 구현 복잡성을 높이지만 클리핑된 목적함수가 핵심 메커니즘입니다.

#### 샘플당 1회 그래디언트 스텝을 사용하는 PPO/GRPO 단순화 (클리핑 없음)

PPO(및 GRPO) 구현은 "샘플당 그래디언트 스텝 수" 하이퍼파라미터가 1인 경우 훨씬 더 간결하게 처리할 수 있습니다.
일반적인 값은 2-4 또는 그 이상입니다.
주요 PPO 또는 GRPO 방정식(@eq:PPO_EQN 참조)에서 "참조" 정책은 이전 파라미터, 즉 완성 또는 행동을 생성하는 데 사용된 것입니다.
따라서 단 하나의 그래디언트 스텝만 수행하면 $\pi_\theta = \pi_{\theta_{\text{old}}}$가 되고, 업데이트 규칙은 다음과 같이 축약됩니다($[]_\nabla$ 표기는 그래디언트 중단을 나타냅니다):

$$J(\theta) = \frac{1}{G}\sum_{i=1}^G \left(\frac{\pi_\theta(a_i|s)}{\left[\pi_{\theta}(a_i|s)\right]_\nabla}A_i - \beta \mathcal{D}_{\text{KL}}(\pi_\theta||\pi_{\text{ref}})\right). $$ {#eq:ppo_1step}

이를 통해 두 번째 정책 그래디언트와 클리핑 로직을 생략할 수 있는 PPO 또는 GRPO 구현이 가능해지며, 최적화기는 표준 정책 그래디언트에 훨씬 더 가까워집니다.


### 예시: 그룹 상대 정책 최적화 (GRPO)

DeepSeekMath 논문은 특히 딥 RL에서의 표준 PPO 적용과 비교할 때, PPO와 다른 GRPO의 일부 구현 세부 사항을 설명합니다 [@shao2024deepseekmath].
예를 들어, RLHF 최적화 내의 KL 패널티(KL 패널티는 보상 모델 없이 검증 가능한 보상으로 추론 모델을 학습시킬 때도 사용된다는 것을 기억하세요)는 보상 함수가 아닌 손실 업데이트에 직접 적용됩니다.
RLHF에 대한 표준 KL 패널티 적용이 $r=r_\theta - \beta \mathcal{D}_{\text{KL}}$로 적용되는 반면, GRPO 구현은 다음과 같습니다:

$$ L = L_{\text{policy gradient}} + \beta * \mathcal{D}_{\text{KL}} $$ {#eq:grpo_loss_kl}

하지만 이를 구현하는 방법은 여러 가지가 있습니다.
전통적으로 KL 거리는 프롬프트 $s$에 대한 완성의 각 토큰에 대해 계산됩니다.
추론 훈련에서는 하나의 프롬프트에서 여러 완성이 샘플링되고, 하나의 배치에 여러 프롬프트가 있으므로, KL 거리는 [B, L, N]의 형태를 갖게 됩니다. 여기서 B는 배치 크기, L은 시퀀스 길이, N은 프롬프트당 완성 수입니다.

이를 종합하여 첫 번째 손실 누적 방식을 사용하면, 의사 코드는 아래와 같이 작성할 수 있습니다.

```python
# B: Batch Size, L: Sequence Length, G: Number of Generations
# Compute grouped-wise rewards # Shape: (B,)
mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1)


# Normalize the rewards to compute the advantages
mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
# Shape: (B*G,)

# Compute advantages
advantages = (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-4)
advantages = advantages.unsqueeze(1)
# Shape: (B*G, 1)

# Compute probability ratio between new and old policies
ratio = torch.exp(new_per_token_logps - per_token_logps)  # Shape: (B*G, L)

# PPO clipping objective
eps = self.cliprange  # e.g. 0.2
pg_losses1 = -advantages * ratio  # Shape: (B*G, L)
pg_losses2 = -advantages * torch.clamp(ratio, 1.0 - eps, 1.0 + eps)  # Shape: (B*G, L)
pg_loss_max = torch.max(pg_losses1, pg_losses2)  # Shape: (B*G, L)

# important to GRPO -- PPO applies this in reward traditionally
# Combine with KL penalty
per_token_loss = pg_loss_max + self.beta * per_token_kl  # Shape: (B*G, L)

# Apply completion mask and compute final loss
loss = ((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
 # Scalar

# Compute core metric for logging (KL, reward, etc. also logged)
with torch.no_grad():
    # Compute clipping fraction
    clip_frac = ((pg_losses2 > pg_losses1).float() * completion_mask).sum() / completion_mask.sum()

    # Compute approximate KL
    approx_kl = (0.5 * ((new_per_token_logps - per_token_logps)**2) * completion_mask).sum() / completion_mask.sum()
```

이 코드를 해석하는 방법에 대한 자세한 내용은 위의 PPO 섹션을 참조하세요. PPO 예시와의 핵심 차이점은 다음과 같습니다:

- **이점 함수 계산**: GRPO는 학습된 가치 함수를 기준선으로 사용하는 대신, 동일한 프롬프트에 대한 생성들 사이의 그룹 상대적 방식으로(평균과 표준편차) 보상을 정규화합니다.
- **가치 네트워크 없음**: GRPO는 가치 모델을 완전히 제거하여 `vf_loss`와 관련된 복잡성을 없앱니다.
- **KL 패널티 위치**: GRPO는 보상에서 KL 패널티를 빼는 대신 손실에 직접 더합니다(이것이 표준 구현이지만, KL 적용 방법에는 더 많은 변형이 존재합니다).

#### RLOO 대 GRPO

RLOO의 이점 함수 업데이트는 GRPO와 매우 유사하며, PPO 스타일의 클리핑 및 KL 패널티 세부 사항과 별개로 고려할 때 알고리즘의 개념적 유사성을 부각시킵니다.
구체적으로, RLOO에서 이점 함수는 GRPO와 매우 유사한 기준선을 기준으로 계산됩니다 -- 동일한 질문에 대한 다른 완성들의 평균 보상 대비 해당 완성의 보상입니다.
간략하게, RLOO 이점 추정치는 다음과 같습니다([TRL](https://github.com/huggingface/trl/blob/bfe20756082488350091352d1cdc19c172e42cd8/trl/trainer/rloo_trainer.py#L433)의 구현에서 확장):

```python
# rloo_k --> number of completions per prompt
# rlhf_reward --> Initially a flat tensor of total rewards for all completions. Length B = N x k
rlhf_reward = rlhf_reward.reshape(rloo_k, -1) #
# Now, Shape: (k, N), each column j contains the k rewards for prompt j.

baseline = (rlhf_reward.sum(0) - rlhf_reward) / (rloo_k - 1)
# baseline --> Leave-one-out baseline rewards. Shape: (k, N)
#  baseline[i, j] is the avg reward of samples i' != i for prompt j.

advantages = rlhf_reward - baseline
# advantages --> Same Shape: (k, N)

advantages = advantages.flatten() # Same shape as original tensor
```

RLOO의 나머지 구현 세부 사항은 정책 그래디언트 구현의 다른 트레이드오프를 따릅니다.

## 보조 주제

정책 그래디언트 알고리즘의 적용을 완벽히 익히기 위해서는 무수히 많은 고려 사항이 있습니다.
여기서는 정책 그래디언트 RL 알고리즘을 성공적으로 배포하는 데 있어 긴 꼬리의 복잡성들을 일부 살펴봅니다.

### 일반화된 이점 추정 (GAE)

일반화된 이점 추정 (GAE)은 정책 그래디언트 알고리즘에서 이점 함수를 계산하는 대안적 방법으로 [@schulman2015high], 편향-분산 트레이드오프를 더 잘 균형 잡습니다.
전통적인 단일 스텝 이점 추정치는 너무 높은 편향을 유발할 수 있으며, 완전한 궤적을 사용하면 높은 분산으로 어려움을 겪을 수 있습니다.
GAE는 다중 스텝 이점 추정치의 지수 가중 평균을 계산하며, $\lambda$ 하이퍼파라미터가 편향-분산 트레이드오프를 제어합니다 -- 단일 스텝 시간차 TD($\lambda=0$)에서 완전 궤적 리턴($\lambda=1$)까지 범위를 가지며; $\lambda=0.95$는 LLM 파인튜닝에서 일반적인 기본값입니다.

이점 추정치는 다양한 형태를 취할 수 있지만, 다음과 같이 $n$ 스텝 이점 추정기를 정의할 수 있습니다(장 초반의 TD 잔차와 유사):

$$
\hat{A}_t^{(n)} = \begin{cases}
r_t + \gamma V(s_{t+1}) - V(s_t), & n = 1 \\
r_t + \gamma r_{t+1} + \gamma^2 V(s_{t+2}) - V(s_t), & n = 2 \\
\vdots \\
r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} + \cdots - V(s_t), & n = \infty
\end{cases}
$$ {#eq:K_STEP_ADV}

여기서 $n$이 짧을수록 각 궤적에 더 많은 학습 능력을 귀속시키기 때문에 분산은 낮지만 편향은 높아집니다 -- 과적합이 발생할 수 있습니다.
GAE는 특정 $n$ 대신 가중 다중 스텝 평균으로 이 공식을 일반화하려고 시도합니다.
시작하기 위해, 예측된 가치의 시간차 (TD) 잔차를 정의해야 합니다.

$$
\delta_t^V = r_t + \gamma V(s_{t+1}) - V(s_t)
$$ {#eq:TD_RESIDUAL}

이를 활용하기 위해, GAE 혼합 파라미터로 또 다른 변수 $\lambda$를 도입합니다. 이것은 추정하고자 하는 미래 이점의 지수 감쇠로 접힙니다:

$$
\begin{array}{l}
\hat{A}_t^{GAE(\gamma,\lambda)} = (1-\lambda)(\hat{A}_t^{(1)} + \lambda\hat{A}_t^{(2)} + \lambda^2\hat{A}_t^{(3)} + \cdots) \\
= (1-\lambda)(\delta_t^V + \lambda(\delta_t^V + \gamma\delta_{t+1}^V) + \lambda^2(\delta_t^V + \gamma\delta_{t+1}^V + \gamma^2\delta_{t+2}^V) + \cdots) \\
= (1-\lambda)(\delta_t^V(1 + \lambda + \lambda^2 + \cdots) + \gamma\delta_{t+1}^V(\lambda + \lambda^2 + \cdots) + \cdots) \\
= (1-\lambda)\left(\delta_t^V\frac{1}{1-\lambda} + \gamma\delta_{t+1}^V\frac{\lambda}{1-\lambda} + \cdots\right) \\
= \sum_{l=0}^{\infty}(\gamma\lambda)^l\delta_{t+l}^V
\end{array}
$$ {#eq:GAE_DFN}

직관적으로, 이것은 이점의 다중 스텝 추정치를 우아하게 평균화하는 데 사용될 수 있습니다.
예시 구현은 아래에 나와 있습니다:

```python
# GAE (token-level) for LM RLHF
#
# B: Batch Size
# L: Length
# Inputs:
#   rewards: (B, L) post-KL per-token rewards
#   values:  (B, L) current V_theta(s_t)
#   done_mask: (B, L) 1.0 at terminal token (EOS or penalized trunc), else 0.0
#   gamma: float (often 1.0),
#   lam (short for lambda): float in [0,1]
#   (Padding beyond terminal should have rewards=0, values=0)
B, L = rewards.shape
advantages = torch.zeros_like(rewards)
next_v = torch.zeros(B, device=rewards.device, dtype=rewards.dtype)
gae = torch.zeros(B, device=rewards.device, dtype=rewards.dtype)

for t in reversed(range(L)):
    not_done = 1.0 - done_mask[:, t]
    delta = rewards[:, t] + gamma * not_done * next_v - values[:, t]
    gae = delta + gamma * lam * not_done * gae
    advantages[:, t] = gae
    next_v = values[:, t]

targets = advantages + values      # y_t for value regression
advantages = advantages.detach()   # for policy loss
```

역방향 루프는 시간차 (TD) 오류 ($\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$)를 누적하는데, 이는 실제 결과가 가치 함수의 예측 대비 얼마나 좋거나 나쁜지를 지수 감쇠 $(\gamma\lambda)^l$와 함께 측정합니다.
종단 토큰에서 `not_done=0`은 미래 상태에서의 부트스트래핑을 방지하고 GAE 누산기를 초기화하여, 각 에피소드의 이점이 독립적으로 계산됩니다(루프가 역방향으로 실행되므로, 종단 토큰은 에피소드 경계에서 지수 가중 누적을 깔끔하게 중지합니다 -- 이는 구현을 패킹 친화적으로 만들어, 하나로 연결된 여러 시퀀스를 올바르게 처리합니다).
최종 `targets`는 이 GAE 루프 외부에서 학습되는 별도의 가치 함수에 대한 회귀 목표로 사용되며, 분리된 `advantages`는 정책 그래디언트에 가중치를 부여합니다 -- 정책 업데이트가 가치 네트워크를 통해 역전파되지 않도록 분리됩니다.
언어 모델의 RLHF에서 $\gamma=1.0$이 일반적인데, 에피소드가 비할인 신용 귀속이 선호되는 짧은 토큰 시퀀스이기 때문입니다(그리고 종종 모두 하나의 토큰 안에 있습니다).

*추가 읽기를 위해서는 [@seita2017gae]를 참조하세요.*

### 이중 정규화

이 장에서 우리는 두 가지 유형의 정규화를 살펴봤습니다. 하나는 스텝 크기 제약이 있는 PPO와 같은 알고리즘에 내장된 것이고, 다른 하나는 최적화 시작 지점을 기준으로 한 KL 발산 기반 거리 패널티입니다.

딥 강화학습의 많은 인기 있는 정책 그래디언트 알고리즘들, PPO와 그 선행 알고리즘들을 포함하여, 에이전트의 학습 과정을 제어할 필요성으로 인해 탄생했습니다.
RLHF에서, 15장 정규화와 3장 훈련 개요에서 광범위하게 논의된 것처럼, 파인튜닝하는 원본 정책으로부터의 거리 패널티를 통한 내장된 정규화 항이 있습니다.
이 관점에서, PPO와 같은 알고리즘(내부 스텝 크기 정규화를 가진)과 REINFORCE(더 단순하며, 특정 하이퍼파라미터 하에서 PPO가 이로 축약됨)와의 차이의 상당 부분은 처음부터 에이전트를 훈련시키는 것에 비해 언어 모델 파인튜닝에서는 훨씬 덜 의미 있습니다.

PPO에서 업데이트의 스텝 크기를 제한하는 목적함수는 [대리 목적함수](https://huggingface.co/blog/deep-rl-ppo#introducing-the-clipped-surrogate-objective)로 알려져 있습니다.
RLHF에서 PPO 정규화가 업데이트에 얼마나 영향을 미치는지 모니터링하려면, 많은 인기 있는 구현에서 클립 분율 변수를 살펴볼 수 있는데, 이는 배치에서 확률 비율이 클리핑 구간 밖으로 떨어지는 샘플의 비율입니다.
이것은 PPO의 정규화기가 활성화되는 빈도에 대한 유용한 대리 지표이지만, 그러한 모든 샘플이 영 그래디언트를 갖는 것은 아닙니다: 대리 목적함수는 클리핑된 분기가 선택될 때만 평탄해지는데, 예를 들어 비율이 $1+\varepsilon$를 초과하는 양의 이점 샘플이나 비율이 $1-\varepsilon$ 미만인 음의 이점 샘플의 경우입니다.

실제로 언어 모델에서 PPO와 GRPO 같은 알고리즘은 종종 배치당 단 하나의 그래디언트 스텝으로 실행되는데, 이는 PPO 고유의 정규화가 전혀 적용되지 않음을 의미합니다(클리핑은 정책이 상당히 변할 때만 배치 내에서 발생할 수 있으므로) 그리고 KL 거리 패널티가 주를 이룹니다.
그러나 이것이 보편적이지는 않습니다. 예를 들어, DAPO는 배치당 16번의 그래디언트 스텝을 사용하며 [@yu2025dapo], Tülu 3은 8B 및 70B 모델에서 배치당 4번의 PPO 업데이트 반복을 사용하지만 훈련 안정성 유지를 위해 405B에서는 1로 줄입니다 [@lambert2024t].

### 추가 읽기

RLHF가 현대 사후 학습의 중심에 자리 잡으면서, 훈련 과정을 개선하기 위해 다른 정책 그래디언트 RL 알고리즘과 일반적인 RL 알고리즘들이 제안되었지만, 최선의 관행을 지배하는 데 있어 중심적인 역할을 하지는 못했습니다.
추가 읽기 예시로는 다음이 있습니다:

- **쌍별 근위 정책 최적화 (P3O; Wu et al., 2023)** [@wu2023pairwise]는 중간 보상 모델을 학습하지 않고 PPO 스타일의 정책 업데이트에서 쌍별 데이터를 직접 사용합니다.
- **소프트 적응형 정책 최적화 (SAPO)** [@gao2025sapo]는 하드 PPO/GRPO 스타일 클리핑을 부드럽고 온도 제어되는 게이팅으로 대체하여, 오프-정책 토큰의 가중치를 낮추면서 온-정책에 가까운 학습 신호를 보존하는 연속적인 신뢰 영역을 목표로 합니다.
- 오프-정책 정책 그래디언트 알고리즘은 **대조 정책 그래디언트 (CoPG)** [@flet2024contrastive](직접 정렬 알고리즘 IPO와 바닐라 정책 그래디언트의 일반화)와 같이 추가적인 비동기 훈련을 가능하게 할 수 있으며, Cohere가 Command A 모델에 사용했습니다 [@cohere2025command].
- **ReMax** [@li2023remax]와 같이 언어 모델을 위해 설계된 REINFORCE 알고리즘의 다른 구현들이 있는데, 이는 보상 모델 추론으로부터의 불확실성 원인을 수용하도록 특별히 설계된 기준선 정규화를 구현합니다.
- Apple Intelligence Foundation Models [@gunter2024apple]나 Kimi k1.5 추론 모델 [@team2025kimi]과 같은 일부 파운데이션 모델들은 **미러 디센트 정책 최적화 (MDPO)** [@tomar2020mirror]의 변형을 사용했습니다. 이 분야의 연구는 여전히 기초를 발전시키고 있지만 [@zhang2025improving], 미러 디센트는 정책 그래디언트 알고리즘이 직접적으로 아닌 최적화 방법입니다. 중요한 점은 기존 RL 인프라와 매우 유사하게 대체된다는 것입니다.
- **분리된 클립 및 동적 샘플링 정책 최적화 (DAPO)**는 긴 추론 궤적이 필요하고 새롭고 덜 활용된 토큰의 확률을 높여야 하는 추론 언어 모델에 더 적합하도록 GRPO에 네 가지 수정을 제안합니다 [@yu2025dapo]. 변경 사항은 다음과 같습니다. 1) 서로 다른 두 클립 하이퍼파라미터 $\varepsilon_\text{low}$와 $\varepsilon_\text{high}$를 사용해 탐색을 늘릴 때 로그 비율의 양수 방향으로 더 큰 단계를 허용한다. 2) 배치에서 보상이 모두 0이거나 모두 1인 샘플을 제거해 학습 신호가 없는 그룹을 제외한다. 3) 앞서 GRPO 구현에서 논의한 것처럼 토큰별 손실을 사용한다. 4) 잘린 답변에서 학습하지 않도록 너무 긴 샘플에 소프트 패널티를 적용한다.
- **가치 기반 증강 근위 정책 최적화 (VAPO)** [@yuan2025vapo]는 DAPO의 최적화(clip-higher, 토큰 수준 정책 그래디언트, 다른 길이 정규화 포함)와 Value-Calibrated PPO [@yuan2025s]의 통찰을 결합한다. 가치 함수를 사전 학습하고 길이 적응형 GAE를 사용함으로써, GRPO와 비교해 가치 기반 방법이 여전히 유망할 수 있음을 보여준다.

## 제안 실험

`code/policy_gradients/`의 동반 구현은 작고 관찰 가능한 RL 실행을 위해 설계되어 있다.
기본 설정은 `reasoning-gym`의 `spell_backward` 절차적 과제에서 `Qwen/Qwen3-1.7B`를 학습한다.
실패와 부분적 진전을 쉽게 확인할 수 있으므로 첫 실험으로 적합하다.

1. **GRPO로 단어 뒤집기 과제를 실행하기.**

   ```bash
   cd code/
   uv run python -m policy_gradients.train --config policy_gradients/configs/grpo.yaml
   ```

   `avg_correctness`, `avg_format`, `avg_binary`를 추적한다.
   가장 먼저 볼 질문은 각 프롬프트 그룹에 대비가 있는지다.
   샘플링된 완성문이 전부 맞거나 전부 틀리면, group-relative 업데이트에는 학습 신호가 거의 없다.

2. **그룹 상대 추정기와 단일 샘플 추정기 비교하기.**
   동일한 시작 설정에 대해 다음을 실행한다:

   ```bash
   cd code/
   uv run python -m policy_gradients.train --config policy_gradients/configs/reinforce.yaml
   uv run python -m policy_gradients.train --config policy_gradients/configs/rloo.yaml
   uv run python -m policy_gradients.train --config policy_gradients/configs/grpo.yaml
   ```

   정확도 신호가 얼마나 빠르게 개선되는지와 손실의 잡음이 얼마나 큰지 비교한다.
   RLOO와 GRPO는 수식만 볼 때보다 프롬프트 내부 기준선의 역할을 훨씬 구체적으로 보여준다.

3. **대비를 조절하는 설정값 탐색하기.**
   `policy_gradients/configs/grpo.yaml`을 복사하고 `num_rollouts`, `temperature`, `data.size`, `format_weight`를 바꾸어 본다.
   작은 `num_rollouts`는 그룹 대비를 줄이고, 너무 낮은 temperature는 샘플을 붕괴시킬 수 있으며, 너무 높은 temperature는 잘못된 형식의 답변을 너무 많이 생성할 수 있다.
   이는 RLVR 레시피가 최적화기를 만지기 전에 샘플링 설정에 많은 노력을 들이는 이유를 보는 가장 단순한 방법이다.

4. **장난감 보상에서 수학 과제로 이동하기.**
   GSM8K식 실험의 경우 새 온라인 RL 환경을 추가하기 전에 `code/reward_models/train_orm.py`와 `code/rejection_sampling/` 예제부터 시작한다.
   좋은 기여 예시는 1B 미만 Qwen 모델에서 실행되고 동일한 그룹 대비 진단을 보고하는 작은 `reasoning-gym` 또는 GSM8K 정책 그래디언트 설정이다.
