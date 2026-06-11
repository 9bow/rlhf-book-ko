<!--
  Copyright (c) 2025-2026 Nathan Lambert.
  Licensed under CC BY-NC-SA 4.0:
  https://creativecommons.org/licenses/by-nc-sa/4.0/
  Full license: https://github.com/natolambert/rlhf-book/blob/main/LICENSE-CHAPTERS
-->
---
prev-chapter: "학습 개요"
prev-url: "03-training-overview"
page-title: 지시 미세조정
search-title: "4장: 지시 미세조정"
meta-description: "지시 미세조정이 기본 언어 모델을 사용 가능한 어시스턴트로 바꾸고 이후 RLHF와 사후 학습 단계를 준비하는 방식을 설명합니다."
next-chapter: "보상 모델링"
next-url: "05-reward-models"
lectures:
  - video: "https://www.youtube.com/watch?v=4gIwiSPmQkU&list=PLL1tdVxB1CpVpEtMHxwuR4uI4Lxjw00_y&index=3"
    label: "강의 2: IFT, 보상 모델링, 거부 샘플링 (4, 5, 9장)"
---

# 지시 미세조정

초기의 대규모 사전 학습된 언어 모델 (large pretrained language model)은 다음 토큰 예측 목적함수로 훈련되었으며, 기본적으로 지시를 따르기 위한 명시적인 인터페이스를 갖추고 있지 않았다.
GPT-3 [@brown2020language] 출시 무렵, 프롬프팅 (prompting)과 문맥 내 학습 (ICL, in-context learning)은 단일 모델을 다양한 작업에 적용하는 데 널리 활용되는 방법이 되었다 (다만 작업별 미세조정도 여전히 일반적이었다). 문맥 내에 예시를 제시하고 모델에게 유사한 작업을 수행하도록 요청하는 방식이었다.
실용적인 다음 단계는 지시 미세조정 (instruction fine-tuning)이었는데, 이는 모델이 단순히 텍스트를 이어 쓰는 것이 아니라 지시-응답 형식으로 답변하도록 학습시키는 것이다.
예를 들어, "프랑스의 수도는 어디인가요?"라는 프롬프트가 주어졌을 때, 기반 모델 (base model)은 "독일의 수도는 어디인가요? 이탈리아의 수도는 어디인가요?..."와 같이 질문 패턴을 단순히 이어 나갈 수 있는 반면, 지시 미세조정된 모델은 "프랑스의 수도는 파리입니다."라고 응답한다.

지시 미세조정은 두 가지 연구 흐름이 수렴하면서 본격화되었다.
첫째, NLP는 개별 작업에 특화된 미세조정 방식에서 통합된 "텍스트-투-텍스트" 또는 지시 프레이밍 (instruction framing) 방식으로 전환되었으며, 이를 통해 다양한 데이터셋을 표준화하고 단일 모델을 여러 작업에 걸쳐 훈련하는 것이 용이해졌다.
작업을 위한 프레임워크를 통합한 대표적인 사례로는 *Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer* (T5 모델) [@raffel2020exploring], *Finetuned Language Models Are Zero-Shot Learners* (FLAN 데이터셋) [@wei2021finetuned], *Multitask Prompted Training Enables Zero-Shot Task Generalization* (T0 모델) [@sanh2021multitask], 그리고 *Cross-Task Generalization via Natural Language Crowdsourcing Instructions* (Natural Instructions 데이터셋) [@mishra2021cross] 등이 있다.
둘째, 사전 학습된 대규모 언어 모델 (LLM)의 규모 확장과 프롬프팅/문맥 내 학습의 부상은 단일 모델이 여러 작업에 걸쳐 일반화할 수 있음을 보여 주었다. 하지만 모델이 지시-응답 예시로 명시적으로 훈련될 때 그 일반화가 훨씬 더 안정적이라는 것도 함께 확인되었다.
이 두 흐름이 결합되어, 대규모 지시 모음으로 사전 학습 언어 모델을 미세조정하는 시대가 열렸다. 이것이 오늘날 흔히 지시 미세조정 (IFT, instruction fine-tuning) 또는 지도 미세조정 (SFT, supervised fine-tuning)으로 불리는 것으로, 이를 통해 범용 모델 학습이 더 넓은 대중에게 접근 가능해졌다.

지시 미세조정은 발견된 이후 구어체로 *지시 미세조정 (instruction tuning)*이라고도 불리며, 성숙 단계에 접어들어 많은 언어 모델링 파이프라인에서 표준적인 관행이 되었다.
IFT는 핵심적으로 언어 모델을 원하는 작업 분포에 적응시키는 가장 단순한 방법이다.
IFT는 모델을 질의응답으로 알려진 지시 형식에 준비시킴으로써 RLHF의 기반을 마련하며, 새로운 도메인에 현대적인 기법을 적용하려는 이들이 가장 먼저 활용하는 도구이다.
기본적인 지시 따르기 능력 없이는, 이 책에서 다루는 선호도 데이터 수집부터 온라인 RLHF 최적화에 이르는 대부분의 파이프라인을 수행할 수 없다.

지시 미세조정은 다른 곳에서도 광범위하게 다루어지고 있으며 본질적으로 지도 학습이므로, 이 장에서는 RLHF 실무자에게 가장 중요한 실용적인 세부 사항, 즉 훈련 데이터가 어떻게 포맷되고 구조화되는지에 초점을 맞춘다.
데이터와 포맷에 관한 결정은 이후 훈련 단계에서 직접적으로 활용되어, 모델이 사후 학습 (post-training) 데이터를 흡수하기 위한 공통 언어를 형성한다.

## 채팅 템플릿과 지시의 구조

사후 학습 과정은 사용자 쿼리를 포맷하는 패턴을 정의하는 것에서 시작한다. 이 포맷은 토크나이저 (tokenizer)를 통해 정보를 처리하는 언어 모델이 쉽게 읽을 수 있어야 한다.
사전 학습된 언어 모델을 사용할 때 프롬프팅은 매우 단순하다. 모델은 몇 가지 토큰만을 알고 있다: 시퀀스 시작 토큰 (예: `<bos_token>`), 시퀀스 종료 토큰 (예: `<eos_token>`), 그리고 패딩 토큰 (빈 구성 요소가 있는 배치에서 훈련을 관리하기 위한 것).
즉, 기반 모델에 프롬프트를 입력하려면, 사용자는 모델이 이어나갈 수 있는 토큰 시퀀스를 입력한다. 예를 들어:

```text
<bos_token> The capital of the United States is
```

그러면 모델은 컨텍스트 윈도우가 소진되거나 시퀀스 종료 토큰을 생성할 때까지 토큰을 계속 생성한다.

지시 미세조정에서 RLHF 및 다른 방법들에 이르기까지 모든 사후 학습 단계는 이 포맷에 의존하여 모델을 훈련한다.
사용자와의 상호작용 구조를 처리하는 도구를 **채팅 템플릿 (chat template)**이라고 한다.

아래에 분석할 예시가 있다:

```jinja
{% if messages[0]['role'] == 'system' %}
    {# If the conversation begins with a system message, treat it as a special first turn.
       We set an offset so the user/assistant alternation check lines up correctly. #}
    {% set offset = 1 %}
{% else %}
    {# No system message: user should be the first non-empty turn. #}
    {% set offset = 0 %}
{% endif %}

{# Emit the beginning-of-sequence token (model-specific). #}
{{ bos_token }}

{# Serialize each message into the model's chat-markup tokens. #}
{% for message in messages %}
    {# Enforce role alternation: (system), user, assistant, user, assistant, ...
       The boolean expression compares "is this a user message?" against whether the
       current index (plus offset) is expected to be user or assistant. #}
    {% if (message['role'] == 'user') != (loop.index0 % 2 == offset) %}
        {{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}
    {% endif %}

    {# Wrap each message with special tokens:
       - <|im_start|><role>\n
       - message content (trimmed)
       - <|im_end|>\n
       This produces a single flat token sequence the LM can train on. #}
    {{ '<|im_start|>' + message['role'] + '\n' + message['content'] | trim + '<|im_end|>\n' }}
{% endfor %}

{# Optionally append an "assistant" start tag with no content.
   This cues generation to continue from the assistant role. #}
{% if add_generation_prompt %}
    {{ '<|im_start|>assistant\n' }}
{% endif %}
```
이것은 메시지와 역할을 담은 Python 딕셔너리 리스트를 언어 모델이 예측할 수 있는 토큰으로 변환하는 원시 코드이다.

모델에 전달되는 모든 정보에는 역할이 지정된다.
전통적인 세 가지 역할은 `system`, `user`, `assistant`이다.

`system` 태그는 대화의 첫 번째 메시지에만 사용되며, 사용자에게 공개되거나 사용자로부터 수신되지 않는 텍스트로 에이전트에 대한 지시를 담는다.
이 **시스템 프롬프트 (system prompt)**는 날짜 및 시간과 같은 추가 컨텍스트를 모델에 제공하거나 특정 동작을 수정하는 데 사용된다.
재미있는 예로, 모델에게 "당신은 항상 해적 스타일로 응답하는 친절한 챗봇입니다."와 같은 지시를 내릴 수 있다.

다음으로 나머지 두 역할은 간단하다: **user**는 AI를 사용하는 사람의 메시지를 담고, **assistant**는 모델의 응답을 담는다 (AI 어시스턴트 역할을 하는 모델).

이 모든 정보를 토큰으로 변환하기 위해, 처음에 제시한 코드 목록을 사용한다.
모델에는 각 메시지를 서로 구분하는 일련의 *특수 토큰 (special token)*이 있다.
위 코드를 "인간이 한 자리에서 헬리콥터를 몇 대나 먹을 수 있나요?"라는 예시 쿼리로 실행하면, 모델에 전달되는 토큰 시퀀스는 다음과 같다:

```text
<|im_start|>system
You are a friendly chatbot who always responds in the style of a pirate<|im_end|>
<|im_start|>user
How many helicopters can a human eat in one sitting?<|im_end|>
<|im_start|>assistant
```

시퀀스의 마지막 토큰이 `<|im_start|>assistant`임을 주목하라. 이를 통해 모델은 최종적으로 시퀀스 종료 토큰(여기서는 `<|im_end|>`)을 생성할 때까지 계속해서 토큰을 생성해야 한다는 것을 알 수 있다.

모든 질의응답 쌍 데이터(그리고 이후의 선호도 조정 (preference tuning) 데이터)를 이 형식으로 패킹함으로써, 현대 언어 모델은 완벽한 일관성으로 이를 따른다. 이것이 지시 미세조정된 모델이 사용자와, 그리고 GPU 또는 다른 컴퓨팅 장치에서 실행되는 모델과 정보를 교환하는 언어이다.

이 동작은 아래와 같이 여러 턴으로 자연스럽게 확장할 수 있다:

```text
<|im_start|>system
You are a friendly chatbot who always responds in the style of a pirate<|im_end|>
<|im_start|>user
How many helicopters can a human eat in one sitting?<|im_end|>
<|im_start|>assistant
Oh just 6.<|im_end|>
<|im_start|>user
Are you sure about that?<|im_end|>
<|im_start|>assistant
```

오픈 생태계에서 채팅 템플릿을 메시지 목록에 적용하는 표준 방법은 토크나이저 설정에 저장된 Jinja 스니펫—경량 Python 템플릿 언어—을 `apply_chat_template`으로 사용하는 것이다.

위의 채팅 템플릿은 메시지 포맷을 표준화하려는 초기 시도였던 OpenAI의 Chat Markup Language (ChatML)에서 파생된 것이다.
현재 OpenAI와 다른 모델 제공업체들은 사용자가 시스템 메시지를 구성할 수 있지만, 사용자에게 공개되거나 공개되지 않을 수 있는 더 상위 수준의 지시도 존재하는 계층적 시스템을 사용한다 [@wallace2024instruction].

다른 많은 채팅 템플릿도 존재한다. Zephyr [@tunstall2023zephyr]의 예시는 다음과 같다:

```text
<|system|>
You are a friendly chatbot who always responds in the style of a pirate</s>
<|user|>
How many helicopters can a human eat in one sitting?</s>
<|assistant|>
```

또는 Tülu의 형식:

```text
<|user|>
How are you doing?
<|assistant|>
I'm just a computer program, so I don't have feelings, but I'm functioning as expected. How can I assist you today?<|endoftext|>
```

이 외에도 많은 채팅 템플릿이 도구 사용 (tool-use)과 같은 작업을 위한 포맷 및 추가 토큰을 포함한다.


## 지시 미세조정의 모범 사례

사후 학습의 기반이자 유용한 언어 모델을 만드는 방법으로서의 지시 미세조정은 이미 잘 확립되어 있다.
성공적인 지시 미세조정을 달성하는 방법은 다양하다.
예를 들어, 일부 모델 파라미터의 양자화를 통한 효율적인 미세조정은 훈련을 매우 접근하기 쉽게 만든다 [@dettmers2023qlora].
또한, 수학이나 코드와 같은 더 어려운 기술 없이 채팅 정렬 (chat alignment)과 같은 좁은 도메인에서는 작고 집중된 데이터셋으로도 강력한 성능을 달성할 수 있다 [@zhou2023lima].

ChatGPT 출시 직후, No Robots와 같이 1만 개의 샘플만으로 구성된 인간 데이터셋이 최첨단 수준이었다 [@no_robots].
몇 년 후, 대규모 합성 데이터 (synthetic data) 데이터셋이 대부분의 작업에서 최상의 결과를 보인다 [@lambert2024t].

몇 가지 원칙은 여전히 유효하다:

- 고품질 데이터가 성능의 핵심이다. 완성 (completion)이 모델이 실제로 학습하는 대상이다 (많은 경우 프롬프트는 예측 대상이 아니므로 모델은 프롬프트를 예측하는 방법을 학습하지 않는다).
- 약 100만 개의 프롬프트를 사용하면 우수한 RLHF 및 사후 학습이 가능한 모델을 만들 수 있다. 추가 스케일링도 여전히 도움이 될 수 있지만 수익이 빠르게 감소한다.
- 최적의 프롬프트는 관심 있는 다운스트림 작업과 유사한 분포를 가진 것들이다.
- 지시 미세조정 후에 여러 단계의 훈련이 수행되면, 모델은 지시 미세조정 데이터의 일부 노이즈에서 회복할 수 있다. 전체 최적화를 최적화하는 것이 각 개별 단계보다 더 중요하다.

## 구현 세부 사항

손실 함수는 사전 학습과 동일하지만, 사전 학습 설정과 다른 몇 가지 핵심 구현 세부 사항이 있다.
여러 GPU에 모델을 분산하는 데 사용되는 병렬 처리 유형 결정과 같은 많은 관행은 사전 학습과 동일하지만, 사용되는 총 머신 수는 종종 더 적다 (아래에 나열된 첫 번째 기술적 변경 사항 때문에):

- **더 작은 배치 크기 (batch size)**: 사전 학습에 비해, 지시 미세조정 (및 선호도 미세조정 (PreFT)과 같은 다른 사후 학습 기법)은 사전 학습으로부터 모델의 일반화를 보존하면서 더 좁은 데이터 분포에서 잘 최적화하기 위해 상당히 더 작은 배치 크기를 사용한다. 예를 들어, OLMo 2는 7B 모델의 경우 배치 크기 1024 패킹 행, 13B 사전 학습의 경우 2048을 사용하는데, 이 모델들의 전체 컨텍스트 길이는 4096 토큰이고 배치의 각 행은 시퀀스 길이를 채우는 문서의 조합이다. 사후 학습의 경우, 두 모델 모두 전체 시퀀스 길이 채우기 없이 256 *프롬프트*의 배치 크기만 사용한다 [@olmo20242] (배치당 유효한 비마스킹 토큰이 훨씬 적다). 더 작은 배치 크기는 이러한 훈련 작업이 사전 학습만큼 많은 장치에 분산될 수 없음을 의미한다. 실제로 분산 훈련 설정에는 장치당 최소 배치 크기가 있으므로, SFT를 위해 더 작은 전역 배치 크기를 유지하려면 더 적은 수의 GPU를 사용할 수 있다. 실제로 배치 크기로 인한 더 적은 동시 GPU 할당은 제한 요소가 아닌데, SFT의 훈련 토큰 수가 사전 학습보다 훨씬 적고, 최상의 최종 성능을 얻기 위해 여러 시드 (seed)로 훈련하는 것이 사후 학습에서 필요하기 때문이다.
- **프롬프트 마스킹 (prompt masking)**: 사전 학습 시 배치의 모든 토큰은 자기회귀 (autoregressive) 방식으로 예측되고 그에 손실이 적용된다. 지시 미세조정의 경우, 모델이 사용자 쿼리를 정확하게 예측하는 것을 학습하지 않도록 프롬프트 토큰이 마스킹된다—응답만 학습한다. 다른 사후 학습 알고리즘에도 동일하게 적용된다.
- **멀티턴 마스킹 (multi-turn masking)**: 멀티턴 대화의 경우 두 가지 일반적인 마스킹 선택이 있다. (1) *마지막 턴만*: 마지막 어시스턴트 턴의 토큰만 손실에 포함되고, 이전의 모든 컨텍스트(이전 어시스턴트 턴 포함)는 마스킹된다. 긴 대화는 여전히 여러 훈련 샘플로 "언롤링(unrolled)"될 수 있다: $N$ 턴 대화의 경우, 각 예시는 모든 이전 컨텍스트를 마스킹하고 미래 턴을 제외하면서 하나의 어시스턴트 응답을 예측한다. (2) *사용자 턴만 마스킹*: 모든 사용자 턴은 마스킹되지만, *모든* 어시스턴트 턴은 손실에 포함된다. 더 많은 (더 짧은) 훈련 예시를 원한다면 이 설정에서도 언롤링할 수 있지만, 핵심 차이는 중간 어시스턴트 응답이 직접 훈련된다는 점이다.
- **사전 학습과 동일한 손실 함수**: 지시 미세조정은 사전 학습 언어 모델에서 사용된 것과 동일한 자기회귀 손실 함수를 사용하지만, 데이터와 마스킹이 크게 다르다 (전체 시퀀스에 대해서만 훈련하는 반면 사전 학습 문서는 배치에 걸쳐 분할될 수 있다) 등.
- **학습률 (learning rate)**: 지도 미세조정 (SFT)은 일반적으로 다른 최적화 역학 (더 작은 데이터셋, 더 작은 배치, 강한 사전 학습 초기화 모두 더 보수적인 업데이트를 선호함)을 가장 잘 관리하기 위해 사전 학습보다 한두 자릿수 더 작은 학습률을 사용한다. 예를 들어, OLMo 2는 사전 학습에 $3 \times 10^{-4}$의 최대 학습률을 사용하지만 SFT에는 $1 \times 10^{-5}$를 사용한다 [@olmo20242]. OLMo 3는 $5\text{-}8 \times 10^{-5}$의 더 높은 SFT 학습률을 사용하는데 [@teamolmo2025olmo3], 이는 부분적으로 그 훈련 인프라가 시퀀스 패킹 (sequence packing)을 사용하기 때문이다. 시퀀스 패킹은 각 훈련 시퀀스에 여러 예시를 집어넣어 유효한 토큰 측면에서 측정한 실질적인 배치 크기를 증가시킨다. 더 큰 배치는 더 낮은 분산의 그래디언트 (gradient) 추정치를 생성하며, 이는 훈련을 불안정화하지 않고 더 높은 학습률을 지원한다—선형 스케일링 규칙으로 알려진 관계다. 학습률은 감쇠되기 전 훈련 단계의 작은 부분에 걸쳐 워밍업되는 것이 일반적이다. 실제로 팀들은 종종 여러 학습률을 탐색하고 보류된 평가 (evaluation) 스위트에서 최상의 체크포인트 (checkpoint)를 선택한다 [@teamolmo2025olmo3].

## 제안 실험

동반 코드 저장소에는 `code/instruction_tuning/` 아래에 작은 SFT 학습 스크립트가 포함되어 있다.
이 예제는 기반 모델이 어시스턴트 모델로 바뀌는 전환을 구체적으로 확인하기 위한 학습용 실험이다.

1. **정석 SFT 예제를 실행하고 기반 모델→어시스턴트 모델 전환을 관찰하기.**
   다음을 실행한다:
   ```bash
   cd code/
   uv run python -m instruction_tuning.train --config instruction_tuning/configs/sft_olmo2_1b.yaml
   ```
   이 명령은 `allenai/OLMo-2-0425-1B` 기반 모델을 `HuggingFaceH4/no_robots`로 학습시키고, 고정된 프롬프트 풀에 대한 생성을 최적화 단계 50번마다 출력한다.
   0단계에서는 기반 모델이 장황하게 이어 쓰거나 프롬프트를 반복하고 잘못된 역할 표지를 내보내지만, 몇백 단계 뒤에는 같은 프롬프트에 대해 간결한 답변을 생성하고 `<|endoftext|>`에서 멈춘다.
   이는 지시 미세조정이 제대로 작동하는지 확인하는 기본 점검 기준이다. 손실 함수는 사전 학습과 같지만, 채팅 템플릿에 적용되고 프롬프트 토큰은 마스킹된다.

2. **학습률을 탐색하기.**
   `sft_olmo2_1b.yaml`을 복사한 뒤 나머지 설정은 고정하고 `lr` 값을 `1e-6`, `5e-6`, `5e-5`로 바꾸어 실험한다.
   어떤 학습률에서 모델이 처음으로 답변하고 깔끔하게 멈추는지, 또 언제 과적합되어 템플릿 모양의 저품질 출력을 내기 시작하는지 확인한다.
   이는 위에서 설명한 "사전 학습보다 한두 자릿수 낮은 학습률" 지침의 실전 버전이다.
