<!--
  Copyright (c) 2025-2026 Nathan Lambert.
  Licensed under CC BY-NC-SA 4.0:
  https://creativecommons.org/licenses/by-nc-sa/4.0/
  Full license: https://github.com/natolambert/rlhf-book/blob/main/LICENSE-CHAPTERS
-->
---
prev-chapter: "추론과 추론 시간 스케일링"
prev-url: "07-reasoning"
page-title: 직접 정렬 알고리즘
search-title: "8장: 직접 정렬 알고리즘"
meta-description: "명시적 보상 모델이나 RL 루프 없이 선호도 목적을 최적화하는 DPO 같은 직접 정렬 알고리즘을 설명합니다."
next-chapter: "거부 샘플링"
next-url: "09-rejection-sampling"
---

# 직접 정렬 알고리즘 (DAAs)

직접 정렬 알고리즘 (Direct Alignment Algorithms, DAAs)은 중간 보상 모델을 훈련하거나 강화학습 최적화기를 사용하지 않고도 동일한 RLHF 목적 함수를 풀도록 모델을 업데이트할 수 있게 해준다.
DAA는 우리가 지금까지 연구해온 동일한 선호도 학습 문제를 (문자 그대로 동일한 데이터로!) 풀어서, 언어 모델을 더 정렬되고, 더 똑똑하며, 사용하기 쉽게 만든다.
보상 모델과 온라인 최적화가 없으면 DAA는 구현이 훨씬 간단해지고, 훈련 중 소비되는 연산이 줄어들며, 실험이 더 쉬워진다.
이 장은 이 알고리즘들을 도출하기 위한 복잡한 수학을 자세히 설명하고, 때로는 지루한 도출 과정이 결국 간단한 구현으로 이어짐을 보여준다.

가장 저명한 DAA이자 언어 모델 정렬의 전체 학문적 운동을 촉발시킨 것은 직접 선호도 최적화 (Direct Preference Optimization, DPO) [@rafailov2024direct]다.
그 핵심에서 DPO는 동일한 제약된 RLHF 목적 함수 (3장 참조)를 풀기 위해 경사 상승법을 사용한다:

$$ \max_{\pi} \mathbb{E}_{x \sim \mathcal{D}}\mathbb{E}_{y \sim \pi(y|x)} \left[r_\theta(x, y)\right] - \beta \mathcal{D}_{\text{KL}}\left(\pi(y|x) \| \pi_{\text{ref}}(y|x)\right)$$ {#eq:review_rlhf}

2023년 5월 출시 이후, 커뮤니티가 DPO와 함께 사용할 적절한 데이터와 하이퍼파라미터를 파악한 짧은 지연 기간 (특히 놀라울 정도로 낮은 학습률) 이후, Zephyr-$\beta$가 2023년 10월에 이를 시작한 것을 계기로 [@tunstall2023zephyr] Llama 3 Instruct [@dubey2024llama], Tülu 2 [@ivison2023camels] 및 3 [@lambert2024t], Nemotron 4 340B [@adler2024nemotron] 등 많은 인기 모델들이 DPO 또는 그 변형을 사용해왔다.
기술적으로, 시퀀스 우도 보정 (Sequence Likelihood Calibration, SLiC-HF)이 최초의 현대적 직접 정렬 알고리즘이었지만 [@zhao2023slic], 여러 요인의 조합으로 인해 널리 채택되지 못했다 (연구 방법의 채택을 되돌리는 것은 항상 까다로운 일이다).

DPO와 DAA의 가장 영향력 있는 부분은 언어 모델 사후 학습 실험의 진입 장벽을 낮추는 것이다---더 적은 연산을 사용하고, 처음부터 구현하기 더 쉬우며, 장난감 예시와 프로덕션 예시 모두에서 작동시키기 더 쉽다.

*이 장 전체에서, $x$는 프롬프트를, $y$는 완성을 나타낸다. 이 표기법은 언어 모델 문헌에서 일반적이며, 방법들은 개별 토큰보다 전체 프롬프트-완성 쌍에서 작동한다.*

## 직접 선호도 최적화 (DPO)

여기서는 DPO가 어떻게 작동하는지에 대한 직관을 설명하고 핵심 방정식을 완전히 재도출한다.

### DPO의 작동 방식

DPO는 표면적으로 RLHF 목적 함수를 풀기 위해 정책을 직접 최적화한다.
이를 위한 손실 함수는 아래에서 도출 과정에서 다시 살펴볼 것인데, 학습된 정책의 선택된 완성과 거부된 완성에 대한 확률이 참조 모델 대비 얼마나 변화했는지를 비교한다.
Bradley-Terry 보상 모델에서 도출된 손실 함수는 다음과 같다:

$$ \mathcal{L}_{\text{DPO}}(\pi_\theta; \pi_{\text{ref}}) = -\mathbb{E}_{(x, y_c, y_r) \sim \mathcal{D}}\left[ \log \sigma\left( \beta \log \frac{\pi_{\theta}(y_c \mid x)}{\pi_{\text{ref}}(y_c \mid x)} - \beta \log \frac{\pi_{\theta}(y_r \mid x)}{\pi_{\text{ref}}(y_r \mid x)} \right) \right] $$ {#eq:dpo_core}

시그모이드 함수 내부에서, 첫 번째 항 $\beta \log \frac{\pi_{\theta}(y_c | x)}{\pi_{\text{ref}}(y_c | x)}$는 정책이 참조 모델 대비 *선택된* 완성의 확률을 얼마나 증가시켰는지를 측정하고, 두 번째 항은 *거부된* 완성에 대해 동일한 것을 한다. 선택된 이동이 거부된 이동을 초과할 때, 즉 정책이 올바른 응답을 선호하도록 학습할 때 손실이 감소한다.

전체에서 $\beta$는 최종 모델과 초기 참조 사이의 KL 발산에 대한 보상 최적화의 균형을 맞추는 하이퍼파라미터다 (즉, 과최적화의 균형을 맞추는 것으로, DPO를 올바르게 사용할 때 중요한 하이퍼파라미터다).
이는 외부 보상 모델을 사용하는 것을 대체하는 DPO 훈련을 위한 암묵적 보상에 의존하는데, 이는 확률의 로그 비율이다:

$$r(x, y) = \beta  \log \frac{\pi_r(y \mid x)}{\pi_{\text{ref}}(y \mid x)}$$ {#eq:dpo_reward}

여기서 $\pi_r(y \mid x)$는 우리가 풀고자 하는 정확한 최적 보상 정책이다.
이는 (5장의 Bradley-Terry 모델 섹션에서 보이듯이) 최적 정책에 대한 Bradley-Terry 보상을 도출함으로써 나온다 (@eq:dpo_opt_policy 참조).
본질적으로, DPO 논문에서 말하듯이, 이 재매개변수화는 "보상 모델이 아닌 최적 정책의 관점에서 인간 선호도 데이터의 확률"을 제공한다---이는 명시적인 보상 모델 학습을 완전히 우회할 수 있음을 의미한다.

최적화기가 줄여야 하는 @eq:dpo_core의 손실을 고려해보자.
여기서, 참조 모델로 정규화된 선택된 응답의 로그 비율이 거부된 응답의 로그 비율보다 클 때 손실이 낮아진다.
실제로, 이는 제시된 데이터의 토큰 시퀀스에 걸친 모델의 로그 확률의 합이다.
따라서 DPO는 선택된 응답과 거부된 응답 사이의 상대적 로그 확률의 격차를 벌린다.

@eq:dpo_reward의 보상으로, 무슨 일이 일어나고 있는지 더 해석하기 위해 손실의 그래디언트를 쓸 수 있다:

$$\nabla_{\theta}\mathcal{L}_{\text{DPO}}(\pi_{\theta}; \pi_{\text{ref}}) = -\beta \mathbb{E}_{(x, y_c, y_r)\sim \mathcal{D}}\left[ w \cdot \left(\nabla_{\theta}\log \pi(y_c \mid x) - \nabla_{\theta}\log \pi(y_r \mid x)\right) \right]$$ {#eq:dpo_gradient}

여기서 $w = \sigma\!\left(r_{\theta}(x, y_r) - r_{\theta}(x, y_c)\right)$이다.

여기서, 그래디언트는 다음을 수행하여 위의 목적 함수를 해결한다:

- 시그모이드 함수 $\sigma(\cdot)$ 내의 첫 번째 항은 보상 추정치가 틀렸을 때 더 높은 0에서 1 사이의 파라미터 업데이트 가중치를 생성한다. 거부된 샘플이 선택된 것보다 선호될 때 가중치 업데이트가 더 커야 한다!
- 둘째, 내부 괄호 $[\cdot]$ 안의 항들은 선택된 응답 $y_c$의 가능성을 높이고 거부된 응답 $y_r$의 가능성을 낮춘다.
- 이 항들은 $\beta$로 가중되는데, 이는 업데이트가 KL 발산 대비 완성의 올바른 순서 지정의 균형을 어떻게 맞출지를 제어한다.


핵심 직관은 DPO가 대응하는 최적 정책을 폐쇄형으로 추출할 수 있는 (@eq:dpo_opt_policy, 경사 하강법과 우리의 ML 도구 덕분에) 암묵적 보상 모델을 피팅한다는 것이다.
DPO 손실은 직접 미분 가능하기 때문에, 보상 모델을 훈련하고 완성을 샘플링하여 채점하는 대리 과정 없이 정확한 그래디언트를 계산하는 것이 간단하다.
종종 오해받는 것은 DPO가 그 핵심에서 보상 모델을 학습하고 있다는 것이며, 따라서 논문의 부제인 *당신의 언어 모델은 몰래 보상 모델이다*가 붙은 것이다.
DPO 목적 함수가 정책을 직접 훈련한다는 것과 혼동하기 쉬우므로, 아래의 도출을 공부하는 것이 완전한 이해에 좋다.

암묵적 보상 모델 학습으로, DPO는 데이터셋의 데이터와 목적 함수의 특정 KL 제약 $\beta$가 주어진 RLHF 목적 함수에 대한 최적 해를 생성한다.
여기서 DPO는 생성이 정책 그래디언트 알고리즘에서처럼 온라인이 아니기 때문에 특정 KL 발산에 대한 정확한 정책을 풀어낸다---선호도 조정을 위한 RL 방법과의 핵심 차이점이다.
많은 면에서, 이는 온라인 RL 방법에 비해 DPO로 $\beta$ 값을 조정하기 더 쉽게 만들지만, 결정적으로 그리고 직관적으로 최적 값은 훈련되는 모델과 그것을 훈련하는 데이터에 따라 달라진다.

선택된 완성과 거부된 완성의 많은 쌍 $y_{chosen} \succ y_{rejected}$로 구성된 각 선호도 데이터 배치에서, DPO는 최적 해를 향해 직접 그래디언트 스텝을 취한다.
이는 정책 그래디언트 방법보다 훨씬 간단하다.

![DPO가 처음 출시되었을 때 RLHF와 선호도 학습을 가장 잘 수행하는 방법에 대한 연구 커뮤니티의 치열한 논쟁을 불러일으켰다. 이 밈은 그 감정을 훌륭하게 포착하는데, 논쟁이 종종 강요되고 과장된 것처럼 느껴졌지만, 입문자부터 최고 연구소의 많은 사람들이 DPO에서 엄청난 이익을 얻고 있었다. DPO 단순성 밈, 크레딧 Tom Goldstein.](images/dpo_meme.jpeg){#fig:dpo-meme}


### DPO 도출

DPO 도출은 두 가지 주요 부분으로 이루어진다.
먼저, 저자들은 이 책 전체에서 사용된 RLHF 목적 함수를 최적으로 풀어내는 정책의 형태를 보여준다.
그 다음, 쌍별 선호도 데이터 (즉, Bradley Terry 모델)로부터 그 해에 도달하는 방법을 보여준다.

#### 1. 최적 RLHF 해 도출

시작을 위해, 이 양을 최대화하고자 함을 나타내면서 RLHF 최적화 목적 함수를 다시 고려해야 한다:

$$ \max_{\pi} \mathbb{E}_{x \sim \mathcal{D}}\mathbb{E}_{y \sim \pi(y|x)} \left[r_\theta(x, y)\right] - \beta \mathcal{D}_{\text{KL}}\left(\pi(y|x) \| \pi_{\text{ref}}(y|x)\right)$$ {#eq:rlhf_opt_eq_repeat}

여기서, 이중 기댓값은 기대 보상을 계산하기 위한 샘플링에만 적용되는데, KL 항은 여전히 분석적 표현이기 때문이다.
먼저, KL 발산의 정의를 전개해보자. $\mathcal{D}_{\text{KL}}(\pi \| \pi_{\text{ref}}) = \mathbb{E}_{y \sim \pi}\left[\log \frac{\pi(y|x)}{\pi_{\text{ref}}(y|x)}\right]$이고, 합의 $\pi(y|x)$ 가중치가 샘플링 분포가 된다는 것을 상기하자.
이제 두 항이 $y \sim \pi(y|x)$에 대한 동일한 기댓값을 공유하므로, 이를 결합할 수 있다:

$$\max_{\pi} \mathbb{E}_{x \sim \mathcal{D}}\mathbb{E}_{y \sim \pi(y|x)}\left[r(x,y)-\beta\log\frac{\pi(y|x)}{\pi_{\text{ref}}(y|x)}\right] $$ {#eq:dpo_deriv_1}

다음으로, 괄호 안의 차이에서 음의 부호를 꺼낸다. 이를 위해 두 항으로 나눈다:

$$ = \max_{\pi}\left(\mathbb{E}_{x \sim \mathcal{D}}\mathbb{E}_{y \sim \pi(y|x)}\left[r(x,y)\right] - \beta\,\mathbb{E}_{x \sim \mathcal{D}}\mathbb{E}_{y \sim \pi(y|x)}\left[\log\frac{\pi(y|x)}{\pi_{\text{ref}}(y|x)}\right]\right) $$ {#eq:dpo_deriv_2}

그런 다음, $-1$을 곱하여 최대화를 최소화로 변환한다:

$$ = \min_{\pi}\left(-\mathbb{E}_{x \sim \mathcal{D}}\mathbb{E}_{y \sim \pi(y|x)}\left[r(x,y)\right] + \beta\,\mathbb{E}_{x \sim \mathcal{D}}\mathbb{E}_{y \sim \pi(y|x)}\left[\log\frac{\pi(y|x)}{\pi_{\mathrm{ref}}(y|x)}\right]\right) $$ {#eq:dpo_deriv_3}

$\beta$로 나누고 재결합한다:

$$ = \min_{\pi}\left(\mathbb{E}_{x \sim \mathcal{D}}\mathbb{E}_{y \sim \pi(y|x)}\left[ \log\frac{\pi(y|x)}{\pi_{\text{ref}}(y|x)} - \frac{1}{\beta}r(x,y) \right]\right) $$ {#eq:dpo_deriv_4}


다음으로, 분배 함수 $Z(x)$를 도입해야 한다:

$$ Z(x) = \sum_y \pi_{\text{ref}}(y|x)\exp\left(\frac{1}{\beta}r(x,y)\right) $$ {#eq:dpo_partition}

분배 함수는 정규화되지 않은 밀도 $\pi_{\text{ref}}(y|x)\exp\left(\frac{1}{\beta}r(x,y)\right)$의 정규화 인자 역할을 하여, 각 고정된 $x$에 대해 $y$에 대한 유효한 확률 함수를 만든다. 이것이 필요한 정확한 이유는 도출을 계속 진행하면서 곧 명확해질 것이다.

이를 대입하면, 중간 변환을 얻는다:

$$ \min_{\pi}\mathbb{E}_{x\sim\mathcal{D}}\mathbb{E}_{y\sim\pi(y|x)}\left[\log\frac{\pi(y|x)}{\frac{1}{Z(x)}\pi_{\text{ref}}(y|x)\exp\left(\frac{1}{\beta}r(x,y)\right)} - \log Z(x)\right] $$ {#eq:dpo_deriv_5}

이것이 어떻게 얻어지는지 보기 위해, @eq:dpo_deriv_4의 최적화에서 대괄호 안의 내부 부분을 고려하자:

$$ \log\frac{\pi(y|x)}{\pi_{\text{ref}}(y|x)} - \frac{1}{\beta}r(x,y) $$ {#eq:dpo_deriv_6}

그런 다음, 양쪽에 $\log Z(x) - \log Z(x)$를 더한다:

$$ = \log\frac{\pi(y|x)}{\pi_{\text{ref}}(y|x)} - \frac{1}{\beta}r(x,y) + \log Z(x) - \log Z(x) $$ {#eq:dpo_deriv_7}

그런 다음, 항들을 그룹화한다:

$$ = \left( \log \frac{\pi(y|x)}{\pi_{\text{ref}}(y|x)} + \log Z(x) \right) - \log Z(x) - \frac{1}{\beta}r(x,y) $$ {#eq:dpo_deriv_8}

$\log(x) + \log(y) = \log(x\cdot y)$ (그리고 $Z$를 분모로 이동)를 사용하면:

$$ = \log \frac{\pi(y|x)}{\frac{1}{Z(x)}\pi_{\text{ref}}(y|x)}- \log Z(x) - \frac{1}{\beta}r(x,y) $$ {#eq:dpo_deriv_9}

다음으로, $\frac{1}{\beta}r(x,y)$를 $\log \exp \frac{1}{\beta}r(x,y)$로 전개하고 동일한 연산을 수행하여 @eq:dpo_deriv_5를 얻는데, 여기서 약간 다시 쓴다:

$$ \min_{\pi}\mathbb{E}_{x\sim\mathcal{D}} \left[ \mathbb{E}_{y\sim\pi(y|x)}\left[\log\frac{\pi(y|x)}{\frac{1}{Z(x)}\pi_{\text{ref}}(y|x)\exp\left(\frac{1}{\beta}r(x,y)\right)} \right] - \log Z(x)\right] $$ {#eq:dpo_deriv_10}

이 최적화 형태로, 우리는 실제로 최적 정책 $\pi^*$를 풀어야 한다.
분배 함수 $Z(x)$를 도입함으로써 $\frac{1}{Z(x)}\pi_{\text{ref}}(y|x)\exp\left(\frac{1}{\beta}r(x,y)\right)$ 항을 $y$에 대한 유효한 확률 분포로 만들었으므로, 내부 기댓값이 사실 고유한 KL 발산임을 인식할 수 있다!

$$ \min_{\pi}\mathbb{E}_{x\sim\mathcal{D}}\left[\mathcal{D}_{\text{KL}} \left(\pi(y|x) \middle\| \frac{1}{Z(x)}\pi_{\text{ref}}(y|x)\exp\left(\frac{1}{\beta}r(x,y)\right) \right) - \log Z(x)\right] $$ {#eq:dpo_deriv_11}

$\log Z(x)$ 항은 최종 답에 의존하지 않으므로 무시할 수 있다. 이는 우리가 학습하는 정책과 분배, $\beta$, 보상, 참조 정책을 관련짓는 형태 사이의 KL 발산만 남긴다.
깁스 불등식은 이것이 두 양이 같을 때, 즉 거리 0에서만 최소화된다는 것을 말해준다!
따라서, 우리는 최적 정책을 얻는다:

$$ \pi^*(y|x) = \pi(y|x) = \frac{1}{Z(x)}\pi_{\text{ref}}(y|x)\exp\left(\frac{1}{\beta}r(x,y)\right) $$ {#eq:dpo_opt_policy}


#### 2. Bradley Terry 모델을 위한 DPO 목적 함수 도출

시작을 위해, 보상 모델링에 관한 5장과 선호도 데이터에 관한 11장에서 인간 선호도의 Bradley-Terry 모델이 다음과 같이 형성됨을 상기하자:

$$p^*(y_1 \succ y_2 \mid x) = \frac{\exp\left(r^*(x,y_1)\right)}{\exp\left(r^*(x,y_1)\right) + \exp\left(r^*(x, y_2)\right)} $$ {#eq:bradley_terry_dpo}

@eq:dpo_opt_policy를 조작하여 최적 보상을 풀 수 있다. 먼저, 양쪽에 로그를 취한다:

$$\log \pi^*(y|x) = \log \left( \frac{1}{Z(x)}\pi_{\text{ref}}(y|x)\exp\left(\frac{1}{\beta}r^*(x,y)\right) \right)$$ {#eq:dpo_reward_deriv1}

$\log(abc) = \log a + \log b + \log c$를 사용하여 우변을 전개한다:

$$\log \pi^*(y|x) = -\log Z(x) + \log \pi_{\text{ref}}(y|x) + \frac{1}{\beta}r^*(x,y)$$ {#eq:dpo_reward_deriv2}

$r^*(x,y)$를 풀기 위해 재배열한다:

$$\frac{1}{\beta}r^*(x,y) = \log \pi^*(y|x) - \log \pi_{\text{ref}}(y|x) + \log Z(x)$$ {#eq:dpo_reward_deriv3}

양쪽에 $\beta$를 곱한다:

$$r^*(x, y) = \beta \log \frac{\pi^*(y \mid x)}{\pi_{\text{ref}}(y \mid x)} + \beta \log Z(x)$$ {#eq:dpo_reward_full}

그런 다음 @eq:bradley_terry_dpo에 표시된 Bradley-Terry 방정식에 보상을 대입하면:

$$p^*(y_1 \succ y_2 \mid x) = \frac{\exp\left(\beta \log \frac{\pi^*(y_1 \mid x)}{\pi_{\text{ref}}(y_1 \mid x)} + \beta \log Z(x)\right)}
{\exp\left(\beta \log \frac{\pi^*(y_1 \mid x)}{\pi_{\text{ref}}(y_1 \mid x)} + \beta \log Z(x)\right) + \exp\left(\beta \log \frac{\pi^*(y_2 \mid x)}{\pi_{\text{ref}}(y_2 \mid x)} + \beta \log Z(x)\right)} $$ {#eq:dpo_loss_deriv0}

지수 표현식을 $e^{a+b}$에서 $e^a e^b$로 분해하고 $e^{\log(Z(x))}$ 항들을 약분하면 다음으로 단순화된다:

$$p^*(y_1 \succ y_2 \mid x) = \frac{\exp\left(\beta \log \frac{\pi^*(y_1 \mid x)}{\pi_{\text{ref}}(y_1 \mid x)}\right)}
{\exp\left(\beta \log \frac{\pi^*(y_1 \mid x)}{\pi_{\text{ref}}(y_1 \mid x)}\right) + \exp\left(\beta \log \frac{\pi^*(y_2 \mid x)}{\pi_{\text{ref}}(y_2 \mid x)}\right)} $$ {#eq:dpo_loss_deriv1}

그런 다음, 분자와 분모에 $\exp\left(-\beta \log \frac{\pi^*(y_1 \mid x)}{\pi_{\text{ref}}(y_1 \mid x)}\right)$를 곱하면:

$$p^*(y_1 \succ y_2 \mid x) = \frac{1}{1 + \exp\left(\beta \log \frac{\pi^*(y_2 \mid x)}{\pi_{\text{ref}}(y_2 \mid x)} - \beta \log \frac{\pi^*(y_1 \mid x)}{\pi_{\text{ref}}(y_1 \mid x)}\right)} $$ {#eq:dpo_loss_deriv2}

마지막으로, $\sigma(x) = \frac{1}{1+e^{-x}}$로서의 시그모이드 함수 정의와 함께:

$$p^*(y_1 \succ y_2 \mid x) = \sigma\left(\beta \log \frac{\pi^*(y_1 \mid x)}{\pi_{\text{ref}}(y_1 \mid x)} - \beta \log \frac{\pi^*(y_2 \mid x)}{\pi_{\text{ref}}(y_2 \mid x)}\right) $$ {#eq:dpo_loss_deriv3}

이것은 최적 정책 $\pi^*$가 주어진 Bradley-Terry 모델 하에서 선호도 데이터의 우도다. 보상 모델링에 관한 5장에서 Bradley-Terry 목적 함수를 우도 최대화, 동등하게는 음의 로그 우도 최소화로 도출했음을 상기하면, 다음과 같은 손실 함수를 얻는다:
$$
\begin{aligned}
\mathcal{L}_{\text{DPO}}(\pi_{\theta}; \pi_{\text{ref}}) &= -\mathbb{E}_{(x,y_c,y_r)\sim\mathcal{D}}\left[ \log p(y_c \succ y_r \mid x)  \right] \\
&= -\mathbb{E}_{(x,y_c,y_r)\sim\mathcal{D}}\left[ \log \sigma\left(\beta \log \frac{\pi_{\theta}(y_c|x)}{\pi_{\text{ref}}(y_c|x)} - \beta \log \frac{\pi_{\theta}(y_r|x)}{\pi_{\text{ref}}(y_r|x)}\right)\right]
\end{aligned}
$${#eq:dpo_loss_deriv4}

이것이 @eq:dpo_core에서 보이는 형태의 DPO 손실 함수다.
DPO 논문에는 실제로는 훨씬 덜 사용되는 Plackett-Luce 모델 하에서의 목적 함수에 대한 추가 도출이 있다 [@rafailov2024direct].

#### 3. Bradley Terry DPO 그래디언트 도출

@eq:dpo_gradient에서 보인 DPO 그래디언트를 모델이 어떻게 학습하는지에 대한 직관을 설명하기 위해 사용했다.
이를 도출하기 위해, 모델 파라미터에 대한 @eq:dpo_loss_deriv4의 그래디언트를 취해야 한다.

$$\nabla_{\theta}\mathcal{L}_{\text{DPO}}(\pi_{\theta}; \pi_{\text{ref}}) = -\nabla_{\theta}\mathbb{E}_{(x,y_c,y_r)\sim\mathcal{D}}\left[ \log \sigma\left(\beta \log \frac{\pi_{\theta}(y_c|x)}{\pi_{\text{ref}}(y_c|x)} - \beta \log \frac{\pi_{\theta}(y_r|x)}{\pi_{\text{ref}}(y_r|x)}\right)\right] $$ {#eq:dpo_grad_0}

시작을 위해, 이를 다시 쓸 수 있다.
시그모이드 함수의 도함수 $\frac{d}{dx} \sigma(x) = \sigma(x)(1-\sigma(x))$, 로그의 도함수 $\frac{d}{dx} \log x = \frac{1}{x}$, 그리고 시그모이드의 성질 $\sigma(-x)=1-\sigma(x)$를 알고 있으므로, 위의 방정식을 재구성할 수 있다.

먼저, $u=\beta \log \frac{\pi_{\theta}(y_c|x)}{\pi_{\text{ref}}(y_c|x)} - \beta \log \frac{\pi_{\theta}(y_r|x)}{\pi_{\text{ref}}(y_r|x)}$ (시그모이드 안의 표현식)로 놓는다.
그러면 다음과 같다:

$$\nabla_{\theta}\mathcal{L}_{\text{DPO}}(\pi_{\theta};\pi_{\text{ref}}) = -\mathbb{E}_{(x, y_c, y_r)\sim \mathcal{D}}\left[\frac{\sigma'(u)}{\sigma(u)}\nabla_{\theta}u\right] $$ {#eq:dpo_grad_2}

이를 전개하고 시그모이드와 로그에 대한 위의 표현식을 사용하면 앞서 소개된 그래디언트를 얻는다:

$$ -\mathbb{E}_{(x,y_c,y_r)\sim\mathcal{D}}\left[\beta\sigma\left(\beta\log\frac{\pi_{\theta}(y_r|x)}{\pi_{\text{ref}}(y_r|x)} - \beta\log\frac{\pi_{\theta}(y_c|x)}{\pi_{\text{ref}}(y_c|x)}\right)\left[\nabla_{\theta}\log\pi(y_c|x)-\nabla_{\theta}\log\pi(y_r|x)\right]\right] $$ {#eq:dpo_grad_3}

## 수치적 우려사항, 약점 및 대안

DPO 알고리즘의 약점을 해결하기 위해 많은 변형들이 제안되었다.
예를 들어, 보상 모델이 생성을 평가할 수 있는 롤아웃 없이, DPO는 모든 선호도 데이터 쌍을 동일한 가중치로 처리한다.
실제로, 선호도 데이터에 관한 11장에서 보듯이, 이진보다 풍부한 레이블로 선호도 데이터를 포착하는 많은 방법들이 있다.
각 쌍을 동등하게 처리하는 것에서 벗어나 최적화를 재균형화하기 위한 여러 알고리즘들이 제안되었다.

- **상대 보상 기반 RL로의 회귀 (REBEL)** 는 더 정확하게 RLHF 문제를 풀기 위해 순수한 쌍별 선호도 데이터가 아닌 선택된 응답과 거부된 응답 사이의 여백으로서 보상 모델의 신호를 추가한다 [@gao2024rebel].
- **보수적 DPO (cDPO)와 아이덴티티 선호도 최적화 (IPO)** 는 선호도 데이터에서의 노이즈를 가정하여 과적합을 해결한다. cDPO는 데이터의 N 퍼센트가 잘못 레이블링되었다고 가정하고 [@rafailov2024direct], IPO는 레이블로부터 직접 최적화하는 것이 아닌 선호도의 확률을 부드럽게 하기 위해 최적화를 변경한다 [@azar2024general]. 실제로, IPO는 선호도 확률을 비선형 함수로 변경하여 Bradley-Terry 가정에서 벗어나며, $\Psi(q) = \log\left(\frac{q}{1-q}\right)$로 나타낸다.
- **오프셋을 가진 DPO (ODPO)** 는 "선호되는 응답과 선호되지 않는 응답의 우도 차이가 오프셋 값보다 크도록 요구한다" [@amini2024direct]---모든 데이터 쌍을 동등하게 취급하지 않지만, 이는 더 어려운 레이블링 환경을 초래할 수 있다.

일부 DPO 변형들은 손실에 소규모 변경을 가하여 학습 신호를 개선하거나, 메모리 사용량을 줄여 적용을 더 효율적으로 만들려 한다.

- **오즈 비율 정책 최적화 (ORPO)** 는 참조 모델에 대한 소규모 패널티와 함께 지시 미세조정 손실과 유사한 선택된 응답 방향의 당김으로 정책 모델을 직접 업데이트한다 [@hong2024reference]. 이 손실 함수의 변경은 참조 모델의 필요성을 제거하여 설정을 단순화한다. ORPO를 보는 가장 좋은 방법은 DPO에서 영감을 받은 것이지, DPO의 파생물이 아니라는 것이다.
- **단순 선호도 최적화 (SimPO)** 는 성능을 향상시키기 위해 DPO 최적화에 소규모 변경을 가하는데, 로그 확률을 합산하는 대신 평균을 내거나 (SimPO) 길이 정규화를 추가한다 [@meng2025simpo].

![DPO에서의 선호도 변위 스케치.](images/dpo_displacement.png){#fig:dpo_issue .center}

DPO에서 *명백한* 핵심 문제 중 하나는 최적화가 선택된 응답과 거부된 응답의 확률 사이의 여백을 증가시키는 방향으로만 유도된다는 것이다.
수치적으로, 모델은 선택된 응답과 거부된 응답 모두의 확률을 줄이지만, @fig:dpo_issue에 나타난 것처럼 *거부된 응답이 더 큰 폭으로 감소한다*.
직관적으로, 이것이 어떻게 일반화되는지는 명확하지 않지만, 연구들은 이것이 다루어지지 않은 행동들의 확률을 증가시킨다고 주장했다---즉, 언어 모델이 생성할 수 있지만 사후 학습 데이터셋의 분포에 없는 토큰들 [@razin2024unintentional] [@ren2024learning].
이 **선호도 변위**를 최적화 과정을 조정하는 Cal-DPO [@xiao2024cal]와 보상 형태를 수정하는 AlphaPO [@gupta2025alphapo] 같은 단순한 방법들로 완화할 수 있다.
실제로 이것의 정확한 영향은 잘 알려져 있지 않지만, 온라인 방법이 일반적인 DPO를 능가할 수 있는 잠재적 이유를 가리킨다.

DPO 유사 방법들이 온라인 (RL 기반) RLHF 방법보다 성능 상한이 낮다고 주장되는 가장 큰 다른 이유는 훈련 신호가 이전 모델이나 다른 모델로부터의 완성에서 나온다는 것이다.
DPO의 온라인 변형들은 훈련 시점에 새로운 완성을 생성하고 선호도 신호를 통합함으로써 이러한 한계를 완화한다. **온라인 DPO** [@guo2024direct]는 현재 모델로부터 생성을 샘플링하는 반면, **판별기 유도 DPO** (D2PO) [@singhal2024d2po]는 즉석에서 새로운 선호도 데이터를 생성하기 위해 보상 모델 재레이블링을 사용하며, 더 많은 변형들이 존재한다.

직접 내시 최적화 (DNO) [@rosset2024direct]나 이진 분류기 최적화 (BCO) [@jung2024binary]와 같은 다른 DAA 변형들의 긴 목록이 있지만, 알고리즘 선택은 초기 모델과 사용된 데이터보다 훨씬 덜 중요하다 [@lambert2024t] [@zhao2024rainbowpo] [@gorbatovski2025differences].

## 구현 세부사항

DPO와 같은 DAA는 정책 그래디언트 최적화기와는 매우 다르게 구현된다.
원래 구현에서 가져온 DPO 손실은 대체로 다음과 같이 요약할 수 있다 [@rafailov2024direct]:

```python
# Log-probability gaps for the policy and the frozen reference model
pi_logratios = policy_chosen_logps - policy_rejected_logps
ref_logratios = reference_chosen_logps - reference_rejected_logps

# Difference of log-ratios: positive when the policy
# shifts probability toward the chosen completion
logits = pi_logratios - ref_logratios

# DPO loss: negative log-sigmoid drives the policy to
# widen the gap between chosen and rejected
losses = -F.logsigmoid(beta * logits)

# Implicit rewards (detached -- used for logging only)
chosen_rewards = beta * (policy_chosen_logps - reference_chosen_logps).detach()
rejected_rewards = beta * (policy_rejected_logps - reference_rejected_logps).detach()
```

이 정보가 모델의 순전파 중에 이미 수집되기 때문에 (참조 모델 추가와 함께), 이는 표준 언어 모델 훈련 스택에서 사용될 수 있다.

대부분의 면에서 DAA는 더 간단하고 삶의 질을 향상시키지만, 다른 고려사항들도 제공한다.

1. **KL 발산은 정적이다**: DPO 및 다른 알고리즘에서 KL 발산은 거리 패널티와 최적화의 균형을 맞추는 $\beta$ 파라미터로 명시적으로 설정된다. 이는 DPO가 데이터가 주어진 RLHF 목적 함수의 *최적* 해를 향해 그래디언트 스텝을 취하기 때문이다---$\beta$ 항에 의해 설정된 해를 정확히 향해 나아간다. 반면, RL 기반 최적화기는 배치와 최근 데이터를 기반으로 스텝을 취한다.
2. **로그 확률 캐싱**: DPO의 단순한 구현은 손실 함수에 대한 편의를 위해 정책 모델과 참조 모델의 순전파를 동시에 수행한다. 하지만 이는 사용되는 메모리를 두 배로 늘리고 GPU 사용량 증가를 초래한다. 이를 피하기 위해, 먼저 훈련 데이터셋에 대한 참조 모델의 로그 확률을 계산한 다음, 배치별 파라미터를 계산하고 업데이트할 때 이를 참조하여 최대 메모리 사용량을 50% 줄일 수 있다.

## 합성 선호도 데이터를 이용한 DAA

요즘 DAA로 선호도 미세조정 (PreFT)을 수행하는 데 사용되는 인기 있는 데이터셋의 대부분은 프론티어 모델이 다른 모델의 출력을 승자 또는 패자로 평가하는 합성 선호도다.
대표적인 예로는 UltraFeedback (이 범주의 첫 번째) [@cui2023ultrafeedback], Tülu 3 (확장된 UltraFeedback 방법론으로 구축) [@lambert2024t], SmolLM 3의 데이터 [@bakouch2025smollm3], 또는 Olmo 3과 함께 출시된 Dolci Pref 데이터셋 [@teamolmo2025olmo3]이 있다.

이러한 데이터셋을 구성하는 모범 사례들은 여전히 발전하고 있다.
2024년 11월 출시된 Tülu 3 및 그 주변의 데이터셋들은 합성 쌍별 선호도 데이터가 미세조정할 모델에서 일부 완성이 생성되는 (더 큰 모델 풀에 혼합되는) 방식으로 "온-정책"이어야 한다는 것을 보여주었다.
이 데이터의 온-정책 특성은 DAA가 모델이 생성하는 올바른 토큰 공간을 최적화하도록 보장했는데---손실 함수들이 지시 미세조정보다 대비적이고 덜 직접적이기 때문이다.
이후 2025년 Olmo 3와 SmolLM 3의 출시와 함께, 다른 연구들은 델타 학습이라는 다른 이론을 지지했는데, 이는 선택된 완성과 거부된 완성 사이의 차이가 완성에 어떤 모델이 사용되는지보다 학습에 더 중요하다고 주장한다 [@geng2025the].
예를 들어, 이 두 모델 모두에서, 선택된 응답은 Qwen 3 32B에서, 거부된 응답은 Qwen 3 0.6B에서 나왔다---두 저자들이 이 쌍을 동시에 독립적으로 개발했다.

전반적으로, DAA로 합성 선호도 데이터에 대해 모델을 훈련하는 것은 구현의 단순성과 강화학습 기반 방법에 비해 강력한 성능으로 인해 대부분의 실무자들이 시작해야 할 곳이다.
합성 선호도 데이터를 광범위하게 사용할 때 완성들 사이를 판단하는 모델의 편향과 같은 다른 소소한 문제들이 있다.
GPT-4와 같은 프론티어 모델들이 길이 편향 [@dubois2024length]과 자신과 일치하는 출력에 대한 선호도 [@panickssery2024llm]를 가진 것으로 알려져 있으므로 (자세한 내용은 12장 참조), 데이터셋의 "선택됨" 섹션에 있는 텍스트가 OpenAI 모델 또는 스타일적으로 유사한 다른 강력한 모델에서 나올 가능성이 약간 더 높다.

이 섹션을 마무리하면서, 이 방법들이 훈련되는 모델의 생성을 어떻게 변경하는지에 대한 직관을 다룰 것이다.
높은 수준에서, 대부분의 DAA는 "선택됨"과 "거부됨" 완성의 확률 사이의 여백을 증가시키기 위해 최적화한다 (일부 덜 인기 있는 알고리즘들은 이 역학을 약간 변경하도록 설계되었지만, 핵심은 유지된다).
이 장의 앞에서 논의한 것처럼 (@fig:dpo_issue 참조), 이는 종종 두 확률이 모두 감소하지만, 거부된 응답이 더 큰 폭으로 감소함을 의미한다.
시퀀스의 각 토큰은 전체 선호도 여백에 얼마나 기여했는지에 따라 다른 그래디언트 (크기와 방향)를 받으며, 이를 통해 최적화기는 결과에 가장 중요한 토큰을 파악할 수 있다.

## DAA 대 RL: 온라인 대 오프라인 데이터

광범위하게, 논쟁은 하나의 질문으로 귀결된다: 언어 모델을 RLHF로 정렬하기 위해 가치 함수, 정책 그래디언트 등을 갖춘 강화학습의 내부 작동이 필요한가?
이것은, 이런 식으로 표현된 대부분의 질문들처럼, 지나치게 단순하다.
물론, 두 방법 모두 잘 확립되어 있지만, 근본적인 차이와 성능 매니폴드가 어디에 있는지 설명하는 것이 중요하다.

여러 보고서들이 정책 그래디언트 기반 및 RL 방법들이 DPO와 그 변형들을 능가한다고 결론지었다.
논쟁은 다른 알고리즘으로 모델을 훈련하되 데이터를 제어하거나 [@ivison2024unpacking] [@xu2024dpo], RL 최적화 루프 내에서 온-정책 데이터의 역할을 연구하는 등 [@tajwar2024preference] 다양한 형태를 취한다.
이 모든 경우에서, DPO 알고리즘들은 약간 뒤처진다.

이 성능 차이에도 불구하고, DAA는 단순성으로 인해 선도적인 모델들에서 여전히 광범위하게 사용된다.
DAA는 훈련 데이터 및 다른 구성에 대한 반복이 빠르게 이루어질 수 있는 제어된 환경을 제공하며, 알고리즘보다 데이터가 훨씬 더 중요한 경우가 많기 때문에 DPO를 사용하는 것이 괜찮을 수 있다.

주로 RL로 훈련되는 추론 모델의 등장으로, 선호도 조정을 위한 RL 사용으로의 추가 투자가 이루어질 것이며, 이는 장기적으로 RL 인프라의 견고성을 향상시키고 인간 피드백에서의 최적화를 위한 DAA와 RL 사이의 이 여백을 굳힐 것이다.

## 제안 실험

`code/direct_alignment/`의 동반 코드는 선호도 데이터에서 DPO와 여러 관련 손실을 학습한다.
이 설정은 오프라인이므로 보상 모델 서버나 롤아웃 루프가 필요 없다.
따라서 선호도 튜닝 실험을 시작하기 가장 접근하기 쉬운 곳이다.

1. **UltraFeedback에서 작은 DPO 실행 학습하기.**

   ```bash
   cd code/
   uv run python -m direct_alignment.train --loss dpo --max_samples 1000
   ```

   `loss`, `accuracy`, `margins`, `chosen_rewards`, `rejected_rewards`를 확인한다.
   핵심 점검 기준은 모델의 샘플 생성이 붕괴하지 않으면서 암묵적 보상 마진이 원하는 방향으로 움직이는지다.

2. **DPO, IPO, 길이 정규화 DPO 비교하기.**

   ```bash
   cd code/
   uv run python -m direct_alignment.train --config direct_alignment/configs/dpo.yaml
   uv run python -m direct_alignment.train --config direct_alignment/configs/ipo.yaml
   uv run python -m direct_alignment.train --config direct_alignment/configs/dpo_norm.yaml
   ```

   마진 스케일과 학습률 민감도를 비교한다.
   IPO의 손실은 DPO와 같은 수치 스케일에 있지 않으므로, 원시 손실만 보지 말고 `accuracy`와 마진 동작을 함께 읽어야 한다.

3. **참조 모델이 없는 변형을 조심스럽게 시도하기.**
   SimPO 또는 ORPO를 해당 설정 파일로 실행한 뒤, 학습 중 로깅되는 생성 샘플을 점검한다.
   이 손실들은 로그확률 스케일링과 학습률에 더 민감하므로 좋은 디버깅 연습이 된다.

   ```bash
   cd code/
   uv run python -m direct_alignment.train --config direct_alignment/configs/simpo.yaml
   uv run python -m direct_alignment.train --config direct_alignment/configs/orpo.yaml
   ```

4. **손실을 바꾸기 전에 데이터를 바꾸기.**
   손실은 고정하고 `--max_samples`, `--max_length`, 또는 선호도 데이터셋을 바꾸어 본다.
   DPO 계열 목적함수 사이의 변경보다 결과가 더 크게 움직인다면, 이는 선호도 튜닝의 중심 주제 중 하나인 "데이터가 보통 작은 알고리즘 차이를 지배한다"는 점을 경험적으로 보여준다.
