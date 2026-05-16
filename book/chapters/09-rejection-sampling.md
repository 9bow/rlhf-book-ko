<!--
  Copyright (c) 2025-2026 Nathan Lambert.
  Licensed under CC BY-NC-SA 4.0:
  https://creativecommons.org/licenses/by-nc-sa/4.0/
  Full license: https://github.com/natolambert/rlhf-book/blob/main/LICENSE-CHAPTERS
-->
---
prev-chapter: "직접 정렬"
prev-url: "08-direct-alignment"
page-title: 거부 샘플링
search-title: "9장: 거부 샘플링"
next-chapter: "선호도의 본질"
next-url: "10-preferences"
lectures:
  - video: "https://www.youtube.com/watch?v=4gIwiSPmQkU&list=PLL1tdVxB1CpVpEtMHxwuR4uI4Lxjw00_y&index=3"
    label: "강의 2: IFT, 보상 모델링, 거부 샘플링 (4, 5, 9장)"
---

# 거부 샘플링

거부 샘플링 (Rejection Sampling, RS)은 선호도 미세조정에서 가장 널리 사용되면서도 가장 적게 문서화된 방법 중 하나다.
많은 저명한 RLHF 논문들이 이를 훈련 파이프라인의 핵심 구성 요소로 사용하지만, 왜 이렇게 잘 작동하는지에 대한 표준 구현이나 설명은 존재하지 않는다.
RS는 훈련 파이프라인의 여러 지점에서 적용될 수 있는데---지시 미세조정 이후, RL 기반 최적화 이후, 또는 심지어 RLVR 이후에도---이를 다재다능하지만 위치 지정이 어려운 도구로 만든다.
문서화가 부족한 특성과 결합하여, 이것이 핵심 최적화 방법들의 끝부분에 여기에 등장하는 이유다.

거부 샘플링은 새로운 후보 완성들을 선별하고, 훈련된 보상 모델을 기반으로 필터링한 다음, 원래 모델을 상위 완성들에 대해서만 미세조정하는 방식으로 작동한다 (지시 미세조정과 동일한 손실 함수).

이 이름은 계산 통계학 [@gilks1992adaptive]에서 유래하는데, 복잡한 분포에서 샘플링하고 싶지만 직접적인 방법이 없는 경우에 해당한다.
이를 해결하기 위해, 더 간단한 분포에서 샘플링하고 그 샘플이 허용 가능한지 확인하는 휴리스틱을 사용한다.
언어 모델에서, 목표 분포는 프롬프트에 대한 고품질 완성이고, 필터는 보상 모델이며, 샘플링 분포는 현재 모델이다.

WebGPT [@nakano2021webgpt], Anthropic의 유용하고 무해한 에이전트 [@bai2022training], OpenAI의 과정 보상 모델 (PRM)에 관한 인기 논문 [@lightman2023let], Llama 2 Chat 모델 [@touvron2023llama], 및 다른 중요한 연구들이 모두 이 기준선을 사용한다. 더 최근의 연구는 이를 직접 공식화했는데, 예를 들어 여러 모달리티의 정렬에 적용하기 위한 RAFT [@dong2023raft]와 거부 샘플링이 다른 선호도 학습 목적 함수들과 어떻게 관련되는지에 대한 원칙적인 개요를 제공하는 통계적 거부 샘플링 최적화 (RSO) [@liu2023statistical]가 있다.

*이 장 전체에서, $x$는 프롬프트를, $y$는 완성을 나타낸다. 이 표기법은 언어 모델 문헌에서 일반적이며, 방법들은 개별 토큰보다 전체 프롬프트-완성 쌍에서 작동한다.*

## 단계별 훈련 과정

거부 샘플링은 전체적으로 몇 가지 단계를 따른다.

0. **프롬프트 및 보상 모델 선택:** 먼저, 다른 훈련 단계에 비해 어떤 프롬프트로 훈련할지 선택해야 한다. 가장 간단한 방법은 첫 번째 SFT/IFT 단계의 모든 프롬프트를 재사용하는 것이지만, 이는 과적합을 유발할 수 있다. 거부 샘플링을 수행하기 전에, 보상 모델도 훈련해야 한다 (자세한 내용은 5장 참조).
1. **시작 체크포인트에서 완성 생성:** 다음으로, 최적화하려는 모델로 선택한 프롬프트에 대한 완성을 생성해야 한다. 여기에는 샘플링 온도, top-p, 최대 시퀀스 길이, 프롬프트당 완성 수 등 많은 설정을 조정하는 것이 포함될 수 있다.
2. **보상 모델로 상위 완성 선택**: 모든 완성은 보상 모델로 순위가 매겨진다. 이 단계에는 프롬프트당 하나의 완성만 유지하는 중복 제거도 포함될 수 있지만, 많은 그러한 설계 선택들은 경험적 절제 연구로 귀결된다.
3. **상위 완성에 대한 SFT:** 거부 샘플링을 마무리하기 위해, 선택된 완성에 대해 시작 체크포인트를 지시 미세조정한다.

거부 샘플링 과정의 시각적 개요는 아래 @fig:rs-overview에 포함되어 있다.

![거부 샘플링 개요.](images/rejection-sampling.png){#fig:rs-overview}

어떤 프롬프트를 사용할지, 어떻게 보상 모델을 선택할지, 거부 샘플링을 어떻게 순서화할지 등에 대한 실제 세부사항은 문헌에 잘 문서화되어 있지 않다.
이 장은 방법에 대한 개요를 제공하고 추가적인 실험은 독자에게 남겨둔다.

### 1. 완성 생성

프롬프트당 여러 후보 완성 집합을 생성하기 위해, $M$개의 프롬프트 집합을 벡터로 정의해보자:

$$X = [x_1, x_2, ..., x_M]$$ {#eq:rs_prompt_vector}

이 프롬프트들은 여러 출처에서 올 수 있지만, 가장 흔하게는 지시 훈련 세트에서 온다.

각 프롬프트 $x_i$에 대해, $N$개의 완성을 생성한다. 이를 행렬로 나타낼 수 있다:

$$Y = \begin{bmatrix}
y_{1,1} & y_{1,2} & \cdots & y_{1,N} \\
y_{2,1} & y_{2,2} & \cdots & y_{2,N} \\
\vdots & \vdots & \ddots & \vdots \\
y_{M,1} & y_{M,2} & \cdots & y_{M,N}
\end{bmatrix}$$ {#eq:rs_completion_matrix}

여기서 $y_{i,j}$는 $i$번째 프롬프트에 대한 $j$번째 완성을 나타낸다.
각 행 $i$는 단일 프롬프트 $x_i$에 해당하며 그것의 $N$개 후보 완성을 포함한다. 각 열 $j$는 모든 프롬프트에 걸쳐 $j$번째로 샘플링된 완성에 해당한다.

### 2. 완성 채점

이제, 이 모든 프롬프트-완성 쌍들을 보상 모델에 통과시켜 보상 행렬을 얻는다.
보상을 행렬 $R$로 나타낼 것이다:

$$R = \begin{bmatrix}
r_{1,1} & r_{1,2} & \cdots & r_{1,N} \\
r_{2,1} & r_{2,2} & \cdots & r_{2,N} \\
\vdots & \vdots & \ddots & \vdots \\
r_{M,1} & r_{M,2} & \cdots & r_{M,N}
\end{bmatrix}$$ {#eq:rs_reward_matrix}

각 보상 $r_{i,j}$는 완성 $y_{i,j}$와 그에 대응하는 프롬프트 $x_i$를 보상 모델 $\mathcal{R}$에 통과시켜 계산된다:

$$r_{i,j} = \mathcal{R}(y_{i,j} \mid x_i)$$ {#eq:rs_reward_computation}

훈련할 상위 완성을 선택하는 여러 방법이 있다.

보상 행렬 $R$을 기반으로 최선의 완성을 선택하는 과정을 공식화하기 위해, 보상 행렬 $R$에 대해 작동하는 선택 함수 $S$를 정의할 수 있다.

#### 프롬프트별 상위

첫 번째 잠재적 선택 함수는 프롬프트당 최대 보상을 취한다.

$$S(R) = [\arg\max_{j} r_{1,j}, \arg\max_{j} r_{2,j}, ..., \arg\max_{j} r_{M,j}]$$ {#eq:rs_selection_per_prompt}

이 함수 $S$는 인덱스 벡터를 반환하는데, 각 인덱스는 $R$의 각 행에서 최대 보상을 가진 열에 해당한다.
이 인덱스들을 사용하여 선택된 완성들을 선택할 수 있다:

$$Y_{chosen} = [y_{1,S(R)_1}, y_{2,S(R)_2}, ..., y_{M,S(R)_M}]$$ {#eq:rs_chosen_completions}


#### 전체 상위 쌍
대안으로, 전체 집합에서 상위 K 프롬프트-완성 쌍을 선택할 수 있다.
먼저, 보상 행렬 R을 단일 벡터로 펼친다:

$$R_{flat} = [r_{1,1}, r_{1,2}, ..., r_{1,N}, r_{2,1}, r_{2,2}, ..., r_{2,N}, ..., r_{M,1}, r_{M,2}, ..., r_{M,N}]$$ {#eq:rs_flattened_rewards}

이 $R_{flat}$ 벡터는 길이가 $M \times N$이며, $M$은 프롬프트 수이고 $N$은 프롬프트당 완성 수다.

이제, $R_{flat}$에서 K개의 가장 높은 값들의 인덱스를 선택하는 선택 함수 $S_K$를 정의할 수 있다:

$$S_K(R_{flat}) = \text{argsort}(R_{flat})[-K:]$$ {#eq:rs_topk_selection}

여기서 $\text{argsort}$는 배열을 오름차순으로 정렬할 인덱스들을 반환하고, 마지막 K 인덱스를 취하여 K개의 가장 높은 값들을 얻는다.

선택된 완성들을 얻기 위해, 이 펼쳐진 인덱스들을 원래 완성 행렬 $Y$로 다시 매핑해야 한다.
대응하는 프롬프트-완성 쌍을 복구하기 위해, 0-인덱스된 펼쳐진 인덱스 $k$를 $i = \lfloor k / N \rfloor + 1$ 및 $j = (k \bmod N) + 1$을 통해 $(i,j)$로 매핑할 수 있다.

#### 선택 예시
다섯 개의 프롬프트와 네 개의 완성을 가진 다음 상황을 고려해보자.
보상을 기반으로 완성을 선택하는 두 가지 방법을 보여줄 것이다.

$$R = \begin{bmatrix}
0.7 & 0.3 & 0.5 & 0.2 \\
0.4 & 0.8 & 0.6 & 0.5 \\
0.9 & 0.3 & 0.4 & 0.7 \\
0.2 & 0.5 & 0.8 & 0.6 \\
0.5 & 0.4 & 0.3 & 0.6
\end{bmatrix}$$ {#eq:rs_example_matrix}

먼저, **프롬프트별**. 직관적으로, 보상 행렬을 다음과 같이 강조할 수 있다:

$$R = \begin{bmatrix}
\textbf{0.7} & 0.3 & 0.5 & 0.2 \\
0.4 & \textbf{0.8} & 0.6 & 0.5 \\
\textbf{0.9} & 0.3 & 0.4 & 0.7 \\
0.2 & 0.5 & \textbf{0.8} & 0.6 \\
0.5 & 0.4 & 0.3 & \textbf{0.6}
\end{bmatrix}$$ {#eq:rs_example_per_prompt}

argmax 방법을 사용하여 각 프롬프트에 대한 최선의 완성을 선택한다:

$$S(R) = [\arg\max_{j} r_{i,j} \text{ for } i \in [1,5]]$$ {#eq:rs_example_selection_formula}

$$S(R) = [1, 2, 1, 3, 4]$$ {#eq:rs_example_selection_result}

이는 다음을 선택함을 의미한다:

- 프롬프트 1에 대해: 완성 1 (보상 0.7)
- 프롬프트 2에 대해: 완성 2 (보상 0.8)
- 프롬프트 3에 대해: 완성 1 (보상 0.9)
- 프롬프트 4에 대해: 완성 3 (보상 0.8)
- 프롬프트 5에 대해: 완성 4 (보상 0.6)

이제, **전체 최선**.
전체 상위 다섯 개의 완성 쌍을 강조해보자.

$$R = \begin{bmatrix}
\textbf{0.7} & 0.3 & 0.5 & 0.2 \\
0.4 & \textbf{0.8} & 0.6 & 0.5 \\
\textbf{0.9} & 0.3 & 0.4 & \textbf{0.7} \\
0.2 & 0.5 & \textbf{0.8} & 0.6 \\
0.5 & 0.4 & 0.3 & 0.6
\end{bmatrix}$$ {#eq:rs_example_top_overall}


먼저, 보상 행렬을 펼친다:

$$R_{flat} = [0.7, 0.3, 0.5, 0.2, 0.4, 0.8, 0.6, 0.5, 0.9, 0.3, 0.4, 0.7, 0.2, 0.5, 0.8, 0.6, 0.5, 0.4, 0.3, 0.6]$$ {#eq:rs_example_flattened}

이제, 다섯 개의 가장 높은 값들의 인덱스를 선택한다:
$$S_5(R_{flat}) = [8, 5, 14, 0, 11]$$ {#eq:rs_example_topk_result}

이를 원래 행렬로 다시 매핑한다:

- 인덱스 8 → 프롬프트 3, 완성 1 (보상 0.9)
- 인덱스 5 → 프롬프트 2, 완성 2 (보상 0.8)
- 인덱스 14 → 프롬프트 4, 완성 3 (보상 0.8)
- 인덱스 0 → 프롬프트 1, 완성 1 (보상 0.7)
- 인덱스 11 → 프롬프트 3, 완성 4 (보상 0.7)

#### 구현 예시

다음은 선택 방법들이 어떻게 구현될 수 있는지 보여주는 코드 스니펫이다.

```python
import numpy as np

x = np.random.randint(10, size=10)
print(f"{x=}")
sorted_indices = np.argsort(x)
x_sorted = x[sorted_indices]
print(f"{x_sorted=}")

# first way to recover the original array
i_rev = np.zeros(10, dtype=int)
i_rev[sorted_indices] = np.arange(10)
np.allclose(x, x_sorted[i_rev])

# second way to recover the original array
np.allclose(x, x_sorted[np.argsort(sorted_indices)])
```

### 3. 미세조정

선택된 완성들로, 현재 버전의 모델에 대해 표준 지시 미세조정을 수행한다.
자세한 내용은 [지시 미세조정 장](https://rlhfbook.com/c/04-instruction-tuning)에서 찾을 수 있다.

## 구현 세부사항

이 훈련을 수행하기 위한 핵심 하이퍼파라미터들은 매우 직관적이다:

- **샘플링 파라미터**: 거부 샘플링은 모델로부터 받은 완성에 직접 의존한다. 거부 샘플링의 일반적인 설정에는 0보다 높은 온도, 예를 들어 0.7에서 1.0 사이의 온도와 top-p 또는 top-k 샘플링과 같은 다른 파라미터의 수정이 포함된다.
- **프롬프트당 완성 수**: 거부 샘플링의 성공적인 구현에는 각 프롬프트당 10개에서 30개 이상의 완성이 포함되었다. 너무 적은 완성을 사용하면 훈련이 편향되거나 노이즈가 생긴다.
- **지시 미세조정 세부사항**: 거부 샘플링 중 지시 미세조정에 대한 명확한 훈련 세부사항은 공개되지 않았다. 초기 지시 미세조정 단계보다 약간 다른 설정을 사용할 가능성이 높다.
- **이기종 모델 생성**: 일부 거부 샘플링 구현은 훈련될 현재 모델뿐만 아니라 여러 모델로부터의 생성을 포함한다. 이를 수행하는 방법에 대한 모범 사례는 확립되지 않았다.
- **보상 모델 훈련**: 사용되는 보상 모델은 최종 결과에 큰 영향을 미친다. 보상 모델 훈련에 대한 더 많은 자료는 [관련 장](https://rlhfbook.com/c/05-reward-models)을 참조하라.

배치 보상 모델 추론을 수행할 때, 토큰화된 완성들을 길이 순으로 정렬하여 배치들이 유사한 길이를 갖도록 할 수 있다.
이는 패딩 토큰에 대해 추론을 실행할 필요성을 제거하고 소규모 구현 복잡성을 대가로 처리량을 향상시킬 것이다.

## 관련: N개 중 최선 샘플링

N개 중 최선 (Best-of-N, BoN)은 거부 샘플링의 근접 관계로, 동일한 생성-채점 절차를 따르지만 선택된 완성에 대해 모델을 **미세조정하지 않는다**.
대신, BoN은 추론 시점에 정적 프롬프트 (또는 프롬프트 집합)에 대한 가능한 최선의 완성을 계산하며, 관련 기법들은 쿼리에 대한 답변을 얻기 위해 추가 연산을 소비하는 채팅 모델의 "Pro" 티어에 자주 사용된다.

N개 중 최선 샘플링은 종종 RLHF 훈련 방법에 대한 기준선으로 포함된다.
BoN이 기본 모델을 수정하지 *않고* 샘플링 기법임을 기억하는 것이 중요하다.
이러한 이유로, PPO와 같은 온라인 훈련 방법과의 BoN 샘플링 비교는 여전히 일부 맥락에서 유효하다.
예를 들어, 다른 정책 대비 BoN 샘플링을 실행할 때 KL 거리를 여전히 측정할 수 있다.

여기서, 단일 프롬프트에 대해 단순한 BoN 샘플링을 사용할 때, 위에서 보인 두 선택 기준 모두 동등함을 보여줄 것이다.

$R$을 $N$개의 완성을 가진 단일 프롬프트에 대한 보상 벡터라 하자:

$$R = [r_1, r_2, ..., r_N]$$ {#eq:rewards_vector}

여기서 $r_j$는 $j$번째 완성에 대한 보상을 나타낸다.

argmax 방법을 사용하여 프롬프트에 대한 최선의 완성을 선택한다:

$$S(R) = \arg\max_{j \in [1,N]} r_j$$ {#eq:selection_function}

$K=1$인 상위-K 방법을 사용하면 동일한 방법으로 귀결되는데, 이는 일반적인 관행이다.

## 제안 실험

`code/rejection_sampling/`의 동반 구현은 완전한 GSM8K 거부 샘플링 파이프라인을 실행한다.
즉 롤아웃을 생성하고, 보상 모델로 채점하고, 학습 하위 집합을 선택하고, 미세조정한 뒤 정확 일치 정확도로 평가한다.
네 개의 설정 파일은 보상 선택 실험과 무작위 대조 실험이 서로 대응되도록 맞춰져 있으므로, 독자는 보상 모델이 실제로 도움이 되는지 물어볼 수 있다.

1. **롤아웃 캐시를 한 번 구축하기.**

   ```bash
   cd code/
   uv run python -m rejection_sampling.preprocess \
       --config rejection_sampling/configs/top_per_prompt.yaml
   ```

   이 명령은 공유 GSM8K 슬라이스에 대한 완성문을 생성하고 채점한다.
   이후 학습 설정 파일들은 생성 및 채점 설정이 바뀌지 않는 한 이 캐시를 재사용한다.

2. **보상 선택을 무작위 대조 실험과 비교하기.**

   ```bash
   cd code/
   uv run python -m rejection_sampling.train \
       --config rejection_sampling/configs/top_per_prompt.yaml
   uv run python -m rejection_sampling.train \
       --config rejection_sampling/configs/random_per_prompt.yaml
   uv run python -m rejection_sampling.train \
       --config rejection_sampling/configs/top_k_overall.yaml
   uv run python -m rejection_sampling.train \
       --config rejection_sampling/configs/random_k_overall.yaml
   ```

   결과는 대응 쌍으로 읽는다.
   `top_per_prompt`는 `random_per_prompt`와, `top_k_overall`은 `random_k_overall`과 비교한다.
   보상으로 선택한 실행이 무작위 기준선을 이기지 못한다면, 해당 슬라이스에서 보상 모델이나 샘플링된 완성문이 유용한 신호를 주지 못하고 있다는 뜻이다.

3. **보상 모델이 선택할 수 있는 폭을 바꾸기.**
   설정 파일 하나를 복사하고 `num_completions_per_prompt`, `temperature`, `top_p`, `selection.top_k`를 바꾸어 본다.
   더 많은 완성문은 가능한 최선의 샘플을 개선할 수 있지만, 보상 모델이 좋은 답과 나쁜 답을 분리할 수 있을 때만 그렇다.

4. **더 작은 정책 모델 시도하기.**
   `model_name`을 더 작은 호환 instruct 모델로 설정하고, `max_train_samples`를 줄인 뒤 같은 대응 쌍을 다시 실행한다.
   이렇게 하면 실험 비용이 낮아지고, 거부 샘플링이 약한 생성을 구제하는지 아니면 이미 좋은 생성들 사이에서 고르는 것뿐인지 더 잘 드러난다.
