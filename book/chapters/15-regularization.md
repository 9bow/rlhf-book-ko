<!--
  Copyright (c) 2025-2026 Nathan Lambert.
  Licensed under CC BY-NC-SA 4.0:
  https://creativecommons.org/licenses/by-nc-sa/4.0/
  Full license: https://github.com/natolambert/rlhf-book/blob/main/LICENSE-CHAPTERS
-->
---
prev-chapter: "과최적화"
prev-url: "14-over-optimization"
page-title: 정규화
search-title: "15장: 정규화"
meta-description: "RLHF와 사후 학습 업데이트가 기본 모델을 망가뜨리지 않고 유용하게 유지되도록 하는 정규화 방법을 설명합니다."
next-chapter: "평가"
next-url: "16-evaluation"
---

# 정규화

이 책에서 우리는 인간 선호도, 검증 가능한 보상, 그리고 다른 가치 있는 신호로부터 학습하기 위해 모델을 수정하는 많은 도구들을 배웠습니다.
우리가 사용하는 모든 방법들은 매우 강력하며, 이전 훈련 단계의 강력하고 일반적인 모델(종종 참조 모델 (reference model)이라고 불림)에 비해 모델이 너무 많이 변하게 할 수 있습니다.
모델이 주어진 보상으로부터 너무 많이 학습하여 분포 외 성능이 저하될 때, 이것을 "과최적화 (over-optimization)"라고 합니다(이전 장에서 논의했듯이).

RLHF 최적화 전반에 걸쳐, 보상 모델의 과최적화를 방지하기 위해 많은 정규화 (regularization) 단계가 사용됩니다.
이러한 맥락에서 과최적화는 무의미한 텍스트를 출력하는 모델처럼 보입니다.
"탈선한" 최적화의 몇 가지 예시로는 모델이 극히 잘못된 답변을 가진 따라갈 수 있는 수학 추론을 출력하거나, 반복적인 텍스트, 언어 전환, 또는 과도한 특수 문자를 출력하는 것이 있습니다.
이 장에서는 모델의 최적화를 제어하는 데 사용되는 다양한 방법들을 다룹니다.

가장 인기 있는 변형은 현재 정책에서 생성된 샘플에 걸쳐 참조 정책까지의 KL 거리로, 집필 시점에 대부분의 RLHF 구현에서 사용됩니다.
"KL 거리"는 KL 발산 (KL divergence) -- 두 확률 분포의 분리를 측정하는 기본 수학적 방법 -- 이 진정한 거리 측도에 필요한 공식 속성을 만족하지 않음에도 불구하고, 훈련 과정 내의 *최적화 거리*를 표현하기 위한 구어적 용어입니다(숫자를 분포 차이의 수치적 측정이 아닌 거리라고 부르는 것이 더 간단합니다).
다른 많은 정규화 기법들이 문헌에 등장했다가 해당 연구 라인의 다음 모델 반복에서 사라지기도 했습니다.
즉, 생성으로부터의 핵심 KL 거리 외의 정규화는 종종 다음 세대에서 단순화될 수 있는 실험적 설정을 안정화하는 데 사용됩니다.
그럼에도 불구하고, RLHF에서 최적화를 제한하는 도구들을 이해하는 것은 중요합니다.

*이 장 전체에서, $x$는 프롬프트를, $y$는 완성을 나타냅니다. 이 표기법은 언어 모델 문헌에서 일반적으로 사용되며, 방법들이 개별 토큰이 아닌 전체 프롬프트-완성 쌍에 대해 작동합니다.*

보상 모델 $r_\theta$가 있는 RLHF 프레임워크에서 사용될 때의 일반적인 공식은 다음과 같습니다:

$$ r = r_\theta - \lambda r_{\text{reg.}} $$ {#eq:rl_start}

참조 구현은 다음과 같습니다:

$$
r = r_\theta - \lambda_{\text{KL}} \mathcal{D}_{\text{KL}} \left( \pi_{\text{RL}}(y \mid x) \, \| \, \pi_{\text{ref}}(y \mid x) \right)
$$ {#eq:kl_standard}

## RL 최적화에서의 KL 발산

수학적 정의는 정의에 관한 부록 A를 참조하세요.
KL 발산은 하나의 확률 분포가 다른 분포로부터 얼마나 멀어졌는지를 측정합니다 -- KL이 0일 때, 두 분포는 동일한 출력을 생성합니다.
다음과 같이 정의됨을 상기하세요:

$$ \mathcal{D}_{\text{KL}}(P || Q) = \sum_{x \in \mathcal{X}} P(x) \log \left(\frac{P(x)}{Q(x)}\right) $$ {#eq:kl_distance_regularization}

RLHF에서, 관심 있는 두 분포는 종종 새로운 모델 버전의 분포, $P(x)$, 그리고 참조 정책의 분포, $Q(x)$입니다.
다른 최적화기는 다른 KL 방향을 사용합니다. 이 책 전반에 걸쳐, 사용되는 가장 일반적인 "KL 패널티"는 참조 정책에 대한 역방향 KL이라고 불립니다. 실제로 이것은 RL 모델에서 토큰을 샘플링하고 참조 모델에서 확률을 계산하는 몬테 카를로 추정치로 줄어듭니다. 직관적으로, 이 역방향 KL은 새로운 모델 $P$ 또는 $\pi_{\text{RL}}$이 원래 참조 모델이 낮은 확률을 할당하는 곳에 상당한 확률 질량을 배치할 때 큰 패널티를 적용하는 수치적 특성을 가집니다.

다른 KL 방향은 ML에서, 예를 들어 일부 RL 알고리즘의 내부 신뢰 영역 계산에서 여전히 자주 사용됩니다. 이 패널티는 직관적으로 업데이트가 $Q$ 또는 $\pi_{\text{ref}}$의 고확률 영역에 확률을 *적용하지 않을 때* 새로운 모델에 패널티를 줍니다. 이것은 증류 (distillation) 또는 행동 복제 (behavioral cloning)에 사용되는 목적에 더 가깝습니다.

### 생성에 대한 참조 모델

KL 패널티는 가장 일반적으로 훈련 중 생성된 토큰과 정적 참조 모델 사이의 거리를 비교하여 구현됩니다.
직관은 훈련하는 모델이 가까이 유지하고 싶은 스타일을 가지고 있다는 것입니다.
이 참조 모델은 가장 자주 지시 미세조정된 (instruction tuned) 모델이지만, 이전 RL 체크포인트일 수도 있습니다.
간단한 대입으로, 우리가 샘플링하는 모델은 $\pi_{\text{RL}}(x)$와 $\pi_{\text{ref}}(x)$가 되며, 위의 @eq:kl_standard 에 표시됩니다(표준 정의에서 종종 $P$와 $Q$로, RL KL 패널티에 적용될 때).
이러한 KL 발산 패널티는 대규모 언어 모델 (LLM)이 대중화되기 훨씬 전에 대화 에이전트에 처음 적용되었으며 [@jaques2017sequence], 사전 학습된 모델의 미세조정을 위한 핵심 기법으로 KL 제어가 빠르게 확립되었습니다 [@jaques2020human].

### 구현 예시

실제로, KL 발산의 구현은 종종 근사화되어 [@schulman2016klapprox] 구현이 훨씬 간단해집니다.
위의 정의에서, 분포 $P$에서 직접 샘플링할 때 KL의 합을 기댓값으로 변환할 수 있습니다(여기서 $x$는 샘플 공간에 대한 일반 랜덤 변수이며, 이 책의 다른 곳에서 사용되는 프롬프트 표기법이 아닙니다).
이 경우, $P$는 현재 훈련 중인 모델(즉, 참조 모델이 아닌)의 생성 분포입니다.
그러면 KL 발산 계산은 다음과 같이 변경됩니다:

$$
\mathcal{D}_{\text{KL}}(P \,||\, Q) = \mathbb{E}_{x \sim P} \left[ \log P(x) - \log Q(x) \right].
$$ {#eq:kl_expectation}

이 방식은 특히 언어 모델 훈련에서 자주 사용되는 로그 확률을 직접 다룰 때 훨씬 간단하게 구현됩니다.

```python
# Step 1: generate() autoregressively samples a full sequence token by token
generated_tokens = model.generate(inputs)

# Step 2: forward() runs a single pass over the sequence to get per-token logits (no sampling)
logits       = model.forward(generated_tokens[:, :-1]).logits
ref_logits   = ref_model.forward(generated_tokens[:, :-1]).logits

# Step 3: Convert logits to log-probabilities
logprobs     = F.log_softmax(logits, dim=-1)
ref_logprobs = F.log_softmax(ref_logits, dim=-1)

# Step 4: Gather the probability each model assigns to the tokens that were actually generated
token_logprobs     = logprobs.gather(-1, generated_tokens[:, 1:].unsqueeze(-1)).squeeze(-1)
ref_token_logprobs = ref_logprobs.gather(-1, generated_tokens[:, 1:].unsqueeze(-1)).squeeze(-1)

# Step 5: Sum to get sequence-level log-probs; their difference approximates KL
seq_logprob     = token_logprobs.sum(dim=-1)
ref_seq_logprob = ref_token_logprobs.sum(dim=-1)

kl_approx = seq_logprob - ref_seq_logprob
kl_full   = F.kl_div(ref_logprobs, logprobs, reduction='batchmean')
```

일부 예시 구현으로는 [TRL](https://github.com/huggingface/trl/blob/5c21de30ae210e4251ead85517ba8dfe3f210e81/trl/trainer/ppo_trainer.py#L1150)과 [Hamish Ivison의 Jax 코드](https://github.com/hamishivi/EasyLM/blob/main/EasyLM/models/llama/llama_train_ppo.py#L278)가 있습니다.


## 암묵적 정규화

이 장의 다른 섹션들은 *명시적* 정규화를 설명합니다: KL 패널티, 사전 학습 그래디언트 (gradient), 그리고 실무자들이 훈련 목적에 의도적으로 추가하는 마진 손실.
증가하는 경험적 연구 결과들은 RL 기반 사후 학습이 *암묵적* 정규화도 제공한다는 것을 보여줍니다 -- 온-정책 최적화 자체의 구조에서 나타나는, 암기와 파국적 망각 (catastrophic forgetting)에 대한 내장된 저항성.
이것은 KL 패널티나 리플레이 버퍼와 같은 RL 훈련을 제어하기 위한 명시적 도구들 없이도, 손실 업데이트의 특성 때문입니다.

### 지도 미세조정은 암기하고, RL은 일반화한다

사후 학습 커뮤니티가 직면한 핵심 질문은 다음과 같습니다: 단일 작업을 훈련할 때, 모델은 보이지 않는 변형으로 전이되는 일반화 가능한 규칙을 학습하는가, 아니면 훈련 분포의 표면적 패턴을 암기하는가?
Chu et al. 2025 [@chu2025sft]는 사후 학습 방법(지도 미세조정 (SFT) 대 RL)이 분포 외 (OOD) 일반화에 미치는 영향을 직접 분리하는 통제된 경험적 연구로 이 질문에 답합니다.
답은 명확합니다: RL은 전이 가능한 규칙을 학습하는 반면, SFT는 훈련 데이터를 암기하고 분포 이동 하에서 무너집니다.

이 연구는 트레이드오프를 이해하기 위해 내장된 규칙 변형이 있는 두 가지 환경을 사용합니다:

- **GeneralPoints**는 모델이 네 장의 트럼프 카드를 받고 수치 값을 연산자(+, -, *, /)로 결합하여 목표 숫자(기본값 24)에 도달해야 하는 산술 카드 게임입니다. OOD 테스트는 페이스 카드 점수 방식을 변경합니다: 훈련은 하나의 규칙(잭, 퀸, 킹 모두 10으로 계산)을 사용하고, 평가는 다른 규칙(잭=11, 퀸=12, 킹=13)을 사용합니다.

- **V-IRL**은 모델이 언어 지시를 따라 도시 거리를 횡단하며 랜드마크를 인식하는 실세계 시각 내비게이션 작업입니다. OOD 이동은 액션 공간을 절대 방향(북, 동)에서 상대 방향(좌, 우)으로 전환합니다.

모든 작업 변형에 걸쳐, RL은 훈련 컴퓨팅이 확장됨에 따라 OOD 성능을 지속적으로 향상시키는 반면, SFT는 분포 내에서 향상됨에도 불구하고 OOD 성능을 지속적으로 *저하*시킵니다.
차이의 크기는 놀랍습니다: OOD 이동이 절대 방향에서 상대 방향 좌표로의 전환인 언어 전용 입력을 사용하는 V-IRL에서, RL은 OOD 단계별 정확도를 80.8%에서 91.8%로 향상시키는 반면, SFT는 80.8%에서 1.3%로 무너집니다.
SFT 모델은 일반화에 실패하는 것 이상입니다: 기본 모델이 이미 가지고 있던 공간 추론을 파괴하고, 명령 구문에서 절대 방향으로의 조회 테이블로 붕괴합니다.

### 실행으로 유지하기: 온-정책 데이터가 망각을 완화한다

이전 섹션은 단일 작업에서 SFT가 암기하는 반면 RL이 일반화함을 보여주었습니다.
Chen et al. 2025 [@chen2025retainingdoingroleonpolicy]는 보완적인 질문을 합니다: 여러 작업을 *순차적으로* 훈련할 때, 모델은 이미 알고 있던 것을 유지하는가?
그들은 RL이 목표 작업에서 비교 가능하거나 더 높은 이득을 달성하면서 SFT보다 훨씬 적게 망각하며, 이 이점을 두 목적이 최적화하는 것의 근본적인 차이로 추적합니다.

두 방법이 왜 그렇게 다르게 동작하는지 이해하기 위해, KL 발산의 렌즈를 통해 그들의 목적을 볼 수 있습니다.
이 섹션에서는 먼저 두 가지 일반적인 사후 학습 방법이 KL 발산의 두 방향으로 매핑될 수 있음을 보이고, 그런 다음 손실 함수로 이것들을 사용하는 수치적 동작이 어떻게 다른 모델 동작으로 변환되는지 설명합니다.

KL 발산은 두 분포 사이의 예상 로그 비율 $\mathbb{E}_{x \sim P}\!\left[\log \frac{P(x)}{Q(x)}\right]$로 정의되며, 두 방향의 로그 차이로 쓸 수 있습니다:

- **순방향 KL (Forward KL)**: $\text{KL}(P \| Q) = \mathbb{E}_{x \sim P}\!\left[\log P(x) - \log Q(x)\right]$
- **역방향 KL (Reverse KL)**: $\text{KL}(Q \| P) = \mathbb{E}_{x \sim Q}\!\left[\log Q(x) - \log P(x)\right]$

여기서 $P$는 목표 분포이고 $Q$는 파라미터 $\theta$로 모델링하는 분포입니다.
핵심 차이는 어느 분포에서 샘플링하느냐입니다: 순방향 KL은 목표(또는 최적) 분포 $P$에서 샘플링하고, 역방향 KL은 우리의 정책 $Q$에서 샘플링합니다.
아래 유도에서, $P$는 목표 $\pi_\star$에 해당하고(SFT를 분석할 때는 훈련 데이터 분포, RL을 분석할 때는 보상 최적 정책), $Q$는 학습된 정책 $\pi_\theta$(우리가 훈련하는 것)에 해당합니다.
SFT는 목표를 먼저 배치합니다 -- $\text{KL}(\pi_\star \| \pi_\theta)$ -- 반면 RL은 순서를 뒤집습니다 -- $\text{KL}(\pi_\theta \| \pi_\star)$ -- 어느 분포에서 샘플링하는지를 변경합니다.
샘플들은 학습할 데이터를 제공합니다. 목적, SFT 또는 RL은, 해당 데이터로부터 모델을 형성합니다.

**SFT $\approx$ 순방향 KL.** 순방향 KL의 정의로 시작합니다:

$$
\text{KL}(\pi_\star \| \pi_\theta) = \mathbb{E}_{(x,y) \sim \mathcal{D}} \left[ \log \pi_\star(y \mid x) - \log \pi_\theta(y \mid x) \right]
$$

로그 차이에 대한 기댓값을 두 항으로 분리하면:

$$
= \mathbb{E}_{(x,y) \sim \mathcal{D}} \left[ \log \pi_\star(y \mid x) \right] - \mathbb{E}_{(x,y) \sim \mathcal{D}} \left[ \log \pi_\theta(y \mid x) \right]
$$

첫 번째 항 $\mathbb{E}\!\left[\log \pi_\star(y \mid x)\right]$는 데이터 분포에만 의존하며 음의 엔트로피 (entropy) $-H(\pi_\star)$와 같습니다 -- $\theta$에 따라 변하지 않는 상수.
두 번째 항 $-\mathbb{E}\!\left[\log \pi_\theta(y \mid x)\right]$는 데이터셋에 대한 음의 로그 우도 (negative log-likelihood)이며, 이것은 표준 SFT 교차 엔트로피 (cross-entropy) 손실 $\mathcal{L}_\text{SFT}(\theta)$입니다. 대입하면:

$$
= \underbrace{-H(\pi_\star)}_\text{const} + \mathcal{L}_\text{SFT}(\theta) \propto \mathcal{L}_\text{SFT}(\theta)
$$ {#eq:sft_forward_kl}

엔트로피 항이 $\theta$에 대해 상수이므로, 두 손실은 동일한 그래디언트와 동일한 최솟값을 공유합니다 -- SFT 손실을 최소화하는 것은 **순방향 KL** 발산 $\text{KL}(\pi_\star \| \pi_\theta)$를 최소화하는 것과 동등합니다.

**RL $\approx$ 역방향 KL.** 표준 KL 정규화 RL 목적으로 시작합니다:

$$
\max_\pi \; \mathcal{J}_\text{RL}(\theta) = \mathbb{E}_{x \sim \mathcal{D},\, y \sim \pi(\cdot \mid x)} \left[ r(x, y) \right] - \beta \cdot \text{KL}\!\left(\pi(\cdot \mid x) \| \pi_\text{ref}(\cdot \mid x)\right)
$$ {#eq:rl_objective_retaining}

$-\beta$를 꺼내면 최대화를 최소화로 변환합니다:

$$
= \min_\pi \; \mathbb{E}_{x \sim \mathcal{D},\, y \sim \pi(\cdot \mid x)} \left[ \log \frac{\pi(y \mid x)}{\pi_\text{ref}(y \mid x)} - \frac{1}{\beta} r(x, y) \right]
$$ {#eq:rl_min_form}

분할 함수 $Z(x) = \sum_y \pi_\text{ref}(y \mid x) \exp\!\left(\frac{1}{\beta} r(x,y)\right)$를 도입하여 보상 기울어진 참조를 유효한 분포로 정규화하고, $\log Z(x)$를 더하고 빼면, 내부 기댓값이 KL 발산이 됩니다:

$$
= \min_\pi \; \mathbb{E}_{x \sim \mathcal{D}} \left[ \text{KL}\!\left(\pi(\cdot \mid x) \;\middle\|\; \frac{1}{Z(x)} \pi_\text{ref}(\cdot \mid x) \exp\!\left(\tfrac{1}{\beta} r(x,y)\right) \right) - \log Z(x) \right]
$$ {#eq:rl_kl_form}

$\log Z(x)$가 $\pi$에 의존하지 않으며 KL 발산이 비음수이고 두 분포가 동일할 때만 0이 되므로, KL은 $\pi$가 보상 기울어진 분포와 같을 때 0에서 최소화됩니다.
따라서 보상 $r(x,y)$ 하에서의 최적 정책은:

$$
\pi_\star(y \mid x) = \frac{1}{Z(x)} \pi_\text{ref}(y \mid x) \exp\!\left(\frac{1}{\beta} r(x,y)\right)
$$ {#eq:optimal_policy_retaining}

이제 역방향 KL과의 연결을 직접 보일 수 있습니다. $\text{KL}(\pi_\theta \| \pi_\star)$를 전개하고 $\log \pi_\star(y \mid x) = \log \pi_\text{ref}(y \mid x) - \log Z(x) + \frac{1}{\beta} r(x, y)$를 대입하면:

$$
\begin{aligned}
\text{KL}(\pi_\theta \| \pi_\star) &= \mathbb{E}_{x \sim \mathcal{D},\, y \sim \pi_\theta(\cdot \mid x)} \left[ \log \pi_\theta(y \mid x) - \log \pi_\star(y \mid x) \right] \\
&= \mathbb{E}_{x \sim \mathcal{D},\, y \sim \pi_\theta(\cdot \mid x)} \left[ \log \pi_\theta(y \mid x) - \log \pi_\text{ref}(y \mid x) + \log Z(x) - \frac{1}{\beta} r(x, y) \right] \\
&= - \frac{1}{\beta} \mathbb{E}_{x,y}\!\left[r(x,y)\right] + \text{KL}\!\left(\pi_\theta(\cdot \mid x) \;\middle\|\; \pi_\text{ref}(\cdot \mid x)\right) + \underbrace{\log Z(x)}_\text{const} \\
&\propto - \frac{1}{\beta} \mathbb{E}_{x,y}\!\left[r(x,y)\right] + \text{KL}\!\left(\pi_\theta(\cdot \mid x) \;\middle\|\; \pi_\text{ref}(\cdot \mid x)\right) \\
&= -\frac{1}{\beta} \mathcal{J}_\text{RL}(\theta)
\end{aligned}
$$

동등하게, RL 목적 $\mathcal{J}_\text{RL}(\theta)$를 최대화하는 것은 **역방향 KL** 발산 $\text{KL}(\pi_\theta \| \pi_\star)$를 최소화하는 것과 같습니다.

이 유도는 SFT와 RL이 근본적으로 다른 목적을 최적화함을 보여줍니다: SFT는 순방향 KL을 최소화하고, RL은 역방향 KL을 최소화합니다.

![순방향 KL (SFT) 대 역방향 KL (RL)의 망각 역학. "오래된" 모드는 이전 지식을, "새로운" 모드는 목표 작업을 나타냅니다. 순방향 KL은 정책을 늘려 목표를 커버하고 오래된 모드에서 질량을 끌어냅니다(오른쪽 위), 반면 역방향 KL은 오래된 모드를 건드리지 않고 새로운 모드를 목표 쪽으로 이동시킵니다(오른쪽 아래). Chen et al. 2025에서, 저자의 허락을 받아.](images/retaining_by_doing_mode_intuition.png){#fig:retaining-mode-intuition}

두 방향의 KL 발산은 다른 최적화 압력을 유발합니다.

순방향 KL은 목표 분포가 모델이 없는 곳에 질량을 가질 때마다 모델에 패널티를 주어 **모드 커버링 (mode covering)**을 장려하는 경향이 있습니다 -- 모델은 목표의 모든 주요 모드를 커버하기 위해 확률을 넓게 분산시킵니다.
이유를 보면: 순방향 KL의 기댓값은 $\pi_\star$ 하에서 취해지므로, 목표가 질량을 가진 영역에 확률을 할당하지 못할 때 모델에 강하게 패널티를 줍니다.

역방향 KL은 모델이 실제로 질량을 배치하는 영역에서만 모델에 패널티를 주어 **모드 탐색 (mode seeking)**을 장려하는 경향이 있습니다: 모델은 다른 것들을 무시하면서 하나의 고확률 모드에 집중할 수 있습니다.
여기서 기댓값은 $\pi_\theta$ -- 모델 자체의 분포 -- 하에서 취해집니다. 따라서 $\pi_\theta(y \mid x) \approx 0$인 영역은 $\pi_\star$가 상당한 질량을 할당하더라도 손실에 거의 기여하지 않습니다.
동시에, 목표가 없는 곳에 질량을 배치할 때 모델에 패널티를 줍니다.

이 구분을 고려하면, SFT가 RL보다 *덜* 망각할 것으로 순진하게 기대할 수 있습니다: 모드 커버링 순방향 KL은 목표의 모든 모드에 걸쳐 질량을 유지하여 오래된 지식을 보존해야 하고, 모드 탐색 역방향 KL은 단일 고보상 모드로 수렴하여 다른 것들을 포기할 수 있습니다.
그러나 반대가 성립합니다.
이 직관은 단일 모드 정책을 가정하지만, 사전 학습된 LLM은 여러 모드를 포함합니다 -- 그리고 다중 모드 분포의 경우 역학이 뒤집힙니다.

두 모드가 있는 정책을 고려해 봅시다: 이전 지식을 나타내는 "오래된" 모드와 목표 작업을 위한 "새로운" 모드(@fig:retaining-mode-intuition).
순방향 KL(SFT)은 목표 분포의 두 모드를 커버하려 하는데, 이는 정책을 늘려 오래된 모드*에서* 확률 질량을 재분배하도록 강제하여 그 형태를 방해하고 망각을 유발합니다.
역방향 KL(RL)은 반면에, 샘플링하는 일부 고보상 영역에만 질량을 배치하면 되므로, 오래된 모드를 전혀 건드리지 않고 샘플링하는 새로운 모드를 목표 쪽으로 이동시켜 이전 지식을 온전히 유지할 수 있습니다.

RL의 모드 탐색 동작 -- 역방향 KL의 구조적 속성 -- 은 모델의 이전 지식의 폭을 보존하고 더 나은 일반화를 가능하게 합니다.

요약하면:

- **SFT (순방향 KL)**: $\text{KL}(\pi_\star \| \pi_\theta)$ -- 샘플은 목표 $\pi_\star$, 즉 인간이 작성한 완성의 고정 데이터셋에서 옵니다. 각 예시에 대해 우리는 묻습니다: 우리 모델 $\pi_\theta$가 이것에 얼마나 많은 확률을 할당하는가? 모델은 절대 아무것도 생성하지 않습니다; 모방하는 법을 배웁니다. 이 모드 커버링 압력은 정책이 질량을 넓게 재분배하도록 강제하여 이전 지식을 방해할 수 있습니다.

- **RL (역방향 KL)**: $\text{KL}(\pi_\theta \| \pi_\star)$ -- 샘플은 우리 자신의 정책 $\pi_\theta$에서 옵니다. 모델이 생성하는 각 완성에 대해 우리는 묻습니다: 이것이 보상 최적 정책 $\pi_\star$에 얼마나 가까운가? 모델이 자신의 생성물에서만 훈련하기 때문에, 업데이트는 모델이 이미 확률 질량을 배치하는 곳에 로컬로 유지됩니다 -- 보상 신호는 어떤 생성물을 강화할지 알려주어, 나머지 분포를 방해하지 않고 $\pi_\star$ 쪽으로 확률을 이동시킵니다.

### RL의 면도날: 온라인 RL이 더 적게 망각하는 이유

이전 섹션은 온-정책 샘플링이 망각에 대한 RL의 저항성을 이끌고 이 메커니즘을 순방향-역방향 KL 역학으로 추적했음을 보여주었습니다.
어떤 주어진 작업에 대해서도, 높은 성능을 달성하는 많은 별개의 정책이 존재합니다.
Shenfeld et al. 2026 [@shenfeld2026rls]은 RL의 일반화에 대한 보완적 관점을 제공하며, 다음을 가정하는 **RL의 면도날 (RL's Razor)** 논제를 도입합니다:

> 새로운 작업에 대한 많은 고보상 솔루션 중에서, RL과 같은 온-정책 방법은 KL 발산으로 측정했을 때 원래 정책에 더 가까이 남는 솔루션 쪽으로 본질적으로 편향됩니다.

![KL 최솟값 솔루션 쪽으로의 편향이 망각을 줄입니다. (왼쪽) 새로운 작업을 해결하는 정책들 중 RL은 기본 모델과 KL이 가장 가까운 것으로 수렴합니다. (오른쪽) 이 KL 편향은 일치하는 새 작업 성능에서 SFT에 비해 더 높은 이전 작업 유지율을 산출합니다. Shenfeld, Pari, and Agrawal 2026에서. 라이선스 CC-BY.](images/rl_razor_motivation.png){#fig:rl-razor-motivation}


저자들은 과거 작업의 망각이 KL 발산으로 측정한 초기 모델로부터 미세조정된 정책이 얼마나 멀어지는지에 직접 비례한다는 것을 발견합니다:

$$
\text{Forgetting} \approx f\!\left(\mathbb{E}_{x \sim \tau}\!\left[\text{KL}\!\left(\pi_0(\cdot \mid x) \| \pi(\cdot \mid x)\right)\right]\right)
$$ {#eq:rl_razor_forgetting}


여러 가지 RL과 SFT 훈련 방식에 걸쳐, 저자들은 망각이 훈련된 정책과 초기 정책 사이의 KL 발산과 강하게 상관관계가 있음을 경험적으로 입증합니다($R^2 = 0.96$), **새로운 작업 데이터를 사용하여 측정했을 때**.
이것은 KL이 이전 작업에서의 보류 데이터가 아닌 *새로운 작업의* 입력 분포에서 측정되었음에도 불구하고, 여전히 과거 작업의 성능 저하를 예측한다는 점에서 놀랍습니다.
실제로, 이것은 기본 정책과 훈련된 정책 사이의 드리프트를 측정함으로써 -- 새로운 전문화된 데이터에서 KL 거리를 측정함으로써 -- 직접 망각을 추정하는 강력한 도구를 제공합니다.

RL 정책에서 더 작은 KL 이동을 유발하는 것이 무엇인지 파악하기 위해, 저자들은 온-정책 대 오프라인 데이터, 그리고 목적이 음의 그래디언트를 포함하는지(보상 기준선 아래로 점수를 받는 샘플에서 RL에 존재하고, 올바른 시연만 강화하는 SFT에는 없는, 잘못된 출력에서 확률을 밀어내는)의 두 축을 따라 RL과 SFT 간의 차이를 분해합니다.
놀랍게도, 온-정책 대 오프라인 데이터가 일반화 성능의 차이를 완전히 설명하는 반면, 음의 그래디언트는 식별 가능한 효과가 없습니다.

직관적으로, 온-정책 방법은 모델이 이미 무시하지 않을 확률로 할당하는 출력을 샘플링하므로, 각 업데이트는 현재 분포 근처에 머물도록 제한됩니다.
반면, SFT는 모델이 현재 생성하는 것으로부터 임의로 멀리 있을 수 있는 고정된 외부 분포에서 훈련하며, 각 그래디언트 단계는 모델 자신의 신념에 관계없이 그 먼 목표 쪽으로 당깁니다.

## 다른 유형의 정규화

사후 학습 문헌 내에서, 많은 주요 모델들이 그들의 설정 내에서 선두 성능에 도달하는 데 도움이 되는 다른 정규화 방법들을 포함합니다.
이 두 가지 예시는 일부 주요 모델들이 안정적인 최적화를 얻기 위해 사후 학습 설정을 어떻게 조작했는지 그림을 그리기 위해 포함된 것이며, 모든 설정에서 명시적으로 작동해야 하는 도구로서가 아닙니다.
더 많은 창의적인 솔루션들이 작동할 수 있고 발견될 것입니다!

### 사전 학습 그래디언트

정규화를 보는 또 다른 방법은 "공개 NLP 데이터셋의 성능 저하를 수정하기 위해" InstructGPT [@ouyang2022training]에서 했듯이, 모델이 가까이 유지하고 싶은 *데이터셋*이 있을 수 있다는 것입니다.
이를 구현하기 위해, 그들은 RLHF의 훈련 목적을 수정합니다.
@eq:rl_start 를 취하면, RLHF에 사용되는 RL 데이터셋의 프롬프트 $x$에서 RL 정책 모델로부터 완성 $y$를 샘플링하여 최적화할 목적 함수로 변환할 수 있습니다:
$$
J(\theta) = \mathbb{E}_{(x,y) \sim \mathcal{D}_{\pi_{\text{RL},\theta}}} \left[ r_{\theta}(y \mid x) - \lambda r_{\text{reg.}} \right]
$$ {#eq:objective_regularization}

그런 다음, 텍스트 일관성을 유지하기 위해 사전 학습 코퍼스(또는 다른 데이터셋)에서 샘플링한 문서에 대한 표준 자기회귀 다음 토큰 예측 손실(사전 학습에서 사용되는)에서 더 높은 확률에 대한 추가 보상을 추가할 수 있습니다:

$$
J(\theta) = \mathbb{E}_{(x,y) \sim \mathcal{D}_{\pi_{\text{RL},\theta}}} \left[ r_{\theta}(y \mid x) - \lambda r_{\text{reg.}} \right] + \gamma \mathbb{E}_{x \sim \mathcal{D}_{\text{pretrain}}} \left[ \log(\pi_{\text{RL},\theta}(x)) \right]
$$ {#eq:objective_pretraining}

최근 연구는 직접 선호도 최적화 (DPO)의 최적화를 균형 잡기 위해 음의 로그 우도 (NLL) 항을 사용할 것을 제안했습니다 [@pang2024iterative].
DPO 손실의 쌍별 특성을 감안할 때, 동일한 손실 수정이 보상 모델 훈련에 적용될 수 있어, 모델이 정확한 텍스트를 예측하도록 제한합니다(작업을 발표하지 않은 연구소의 소문).

최적화는 DPO에 대한 수정으로 따릅니다.
$$\mathcal{L}_{\text{DPO+NLL}} = \mathcal{L}_{\text{DPO}}(c_i^w, y_i^w, c_i^l, y_i^l \mid x_i) + \alpha \mathcal{L}_{\text{NLL}}(c_i^w, y_i^w \mid x_i)
$$ {#eq:dpo_nll}

$$
= -\log \sigma \left( \beta \log \frac{P_\theta(c_i^w, y_i^w \mid x_i)}{P_{\text{ref.}}(c_i^w, y_i^w \mid x_i)} - \beta \log \frac{P_\theta(c_i^l, y_i^l \mid x_i)}{P_{\text{ref.}}(c_i^l, y_i^l \mid x_i)} \right) - \alpha \frac{\log P_\theta(c_i^w, y_i^w \mid x_i)}{|c_i^w| + |y_i^w|},
$$ {#eq:dpo_nll_expanded}

여기서 $P_{\theta}$는 훈련 가능한 정책 모델이고, $P_{\text{ref.}}$는 고정된 참조 모델(종종 SFT 체크포인트)이며, $(c_i^w, y_i^w)$와 $(c_i^l, y_i^l)$은 프롬프트 $x_i$에 대한 승리 및 패배 완성을 나타냅니다.
첫 번째 항은 표준 DPO 로지스틱 손실입니다: 로그 가능도 비율의 차이 $\log \tfrac{P_{\theta}}{P_{\text{ref.}}}$를 사용하여 승리와 패배 사이의 마진을 증가시키며, $\beta$는 이 선호도 신호가 참조로부터 얼마나 강하게 당기는지를 제어합니다.
두 번째 항은 승리 완성에 대한 길이 정규화된 음의 로그 우도 패널티로, $\alpha$로 가중치를 줍니다. 이것은 선호된 텍스트를 거부된 샘플보다 상대적으로 더 나을 뿐만 아니라 절대적인 언어 모델링 의미에서 높은 가능도를 유지하는 데 도움을 줍니다.

### 마진 기반 정규화

최적화를 제어하는 것은 RLHF 스택의 다른 부분에서 덜 명확하게 정의되어 있습니다.
대부분의 보상 모델은 표준 대조 손실 함수 외의 정규화가 없습니다.
직접 정렬 알고리즘 (Direct Alignment Algorithms)은 $\beta$ 파라미터를 통해 KL 발산에 대한 정규화를 다르게 처리합니다([직접 정렬 챕터](https://rlhfbook.com/c/08-direct-alignment) 참조).

Llama 2는 보상 모델 훈련을 위한 마진 손실을 제안했습니다 [@touvron2023llama]:

$$
\mathcal{L}(\theta) = - \log \left( \sigma \left( r_{\theta}(y_c \mid x) - r_{\theta}(y_r \mid x) - m(y_c, y_r) \right) \right)
$$ {#eq:margin_loss}

여기서 $m(y_c, y_r)$은 두 주석자의 평점 간 델타의 수치적 차이를 나타내는 두 데이터 포인트 $y_c$와 $y_r$ 사이의 마진입니다.
이는 주석자가 출력을 수치 척도로 평가하거나 [리커트 척도 (Likert scale)](https://en.wikipedia.org/wiki/Likert_scale)와 같은 정량화된 순위 방법을 사용하여 달성됩니다.

보상 마진은 직접 정렬 문헌에서 많이 사용되었습니다. 예를 들어 보상 가중 DPO, DPO 손실을 따라 보상 모델 점수를 업데이트 규칙에 통합하는 "보상 인식 선호도 최적화 (Reward-aware Preference Optimization, RPO)" [@adler2024nemotron], 또는 회귀 손실 공식에서 보상 델타 가중치를 갖는 REBEL [@gao2024rebel] 등이 있습니다.
