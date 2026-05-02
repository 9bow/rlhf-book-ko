<!--
  Copyright (c) 2025-2026 Nathan Lambert.
  Licensed under CC BY-NC-SA 4.0:
  https://creativecommons.org/licenses/by-nc-sa/4.0/
  Full license: https://github.com/natolambert/rlhf-book/blob/main/LICENSE-CHAPTERS
-->
---
prev-chapter: "합성 데이터 및 CAI"
prev-url: "12-synthetic-data"
page-title: 도구 사용
search-title: "13장: 도구 사용"
next-chapter: "과최적화"
next-url: "14-over-optimization"
---

# 도구 사용 및 함수 호출

언어 모델이 도구를 사용하는 것은 모델의 능력을 확장하는 자연스러운 방법입니다. 특히 외부 도구가 정보를 갖고 있는 높은 정밀도가 요구되는 작업이나, 복잡한 웹 시스템과 상호작용해야 하는 에이전트에 유용합니다.
도구 사용은 언어 모델이 훈련을 통해 습득해야 하는 기술이며, 이 책에서 소개하는 RLHF 및 여러 방법들로 이를 정제할 수 있습니다.
다음과 같은 사용자 질문을 생각해 보겠습니다:

> **사용자**: 오늘 대통령은 누구인가요?

도구 없는 언어 모델은 사전 학습 데이터의 지식 마감일 때문에 이 질문에 답하기 어렵지만, 단 한 번의 검색 쿼리로 쉽게 접근할 수 있는 정보입니다.
다른 예시도 생각해 봅시다:

> **사용자**: 다운로드 폴더에 있는 모든 arXiv 논문을 논문 날짜가 표시된 이름으로 ~/research/ 디렉터리로 이동해 주세요.

이것은 모델 가중치만으로는 시도조차 할 수 없는 작업입니다 -- 도구 사용은 언어 모델이 훨씬 더 광범위한 작업을 처리할 수 있게 해줍니다.

본격적인 내용에 들어가기 전에, 흔히 혼용되는 관련 용어들을 구분해 두는 것이 유용합니다:

- **도구 사용 (Tool use)**: 모델이 구조화된 요청(도구 이름과 인수)을 출력하면, 오케스트레이터가 도구를 실행하고, 결과가 컨텍스트에 추가되며, 모델이 생성을 계속합니다.
- **함수 호출 (Function calling)**: 인수가 선언된 스키마(보통 JSON Schema)를 따라야 하는 도구 사용으로, 신뢰할 수 있는 파싱과 검증이 가능합니다.
- **코드 실행 (Code execution)**: "도구"가 코드 인터프리터(예: Python)인 도구 사용의 특수한 경우로, 결과가 도구 출력으로 반환됩니다.

AI 모델은 특정 엔드포인트를 트리거하기 위한 특수 토큰을 출력함으로써 외부 도구를 사용합니다.
이는 특정 장소의 날씨를 반환하는 함수처럼 매우 구체적인 도구부터, 복잡한 행동의 기본 구성 요소로 작동하는 코드 인터프리터나 검색 엔진에 이르기까지 다양합니다.
첫 번째 예시는 언어 모델이 과거 데이터로 훈련된 고정된 가중치를 보완하기 위해 더 최신 정보가 필요한 경우를 보여주었지만, 코드 실행과 같은 도구도 있습니다. 코드 실행은 언어 모델의 확률적·생성적 특성을 우회하여 정확한 답변을 반환할 수 있게 해줍니다.
예를 들어 pi를 50자리까지 근사값으로 출력하는 작업을 생각해 보겠습니다(기억에서 암송하다가 환각이 발생하는 위험 없이).
도구를 갖춘 언어 모델은 다음과 같이 할 수 있습니다:

```text
<code>
from decimal import Decimal, getcontext
getcontext().prec = 60

def compute_pi():
    # Chudnovsky algorithm for computing pi
    C = 426880 * Decimal(10005).sqrt()
    K, M, X, L, S = 0, 1, 1, 13591409, Decimal(13591409)
    for i in range(1, 100):
        M = M * (K**3 - 16*K) // ((i)**3)
        K += 12
        L += 545140134
        X *= -262537412640768000
        S += Decimal(M * L) / X
    return C / S

print(str(compute_pi())[:52])
</code>

<output>
3.14159265358979323846264338327950288419716939937510
</output>
```

이 장에서는 현대 언어 모델에서 도구 사용의 기원, 기본 원리와 형식, 그리고 주요 모델에서 도구를 효과적으로 활용하기 위한 현재의 트레이드오프를 살펴봅니다.

"도구 사용"이라는 용어의 정확한 기원은 명확하지 않지만, 이 개념의 기원은 RLHF가 확산된 ChatGPT 이후 세계보다 훨씬 앞서 있습니다.
2015년경의 초기 사례들은 현대 언어 모델 이전 시스템을 구축하려 했습니다. 예를 들어 Neural Programmer-Interpreters(NPI) [@reed2015neural]는 "프로그램을 표현하고 실행하는 방법을 학습하는 순환적이고 합성적인 신경망"이었습니다.
언어 모델이 더 널리 쓰이게 되면서, 많은 하위 분야에서 성능을 향상하기 위해 외부 기능과의 통합을 활용하기 시작했습니다.
가중치 외부의 정보를 얻기 위해 많은 연구자들이 검색 증강 생성 (Retrieval Augmented Generation) [@lewis2020retrieval] 또는 웹 브라우징 [@nakano2021webgpt]을 사용했습니다.
이후 언어 모델을 프로그램 [@gao2023pal] 또는 도구 [@parisi2022talm]와 통합하는 연구도 등장했습니다.

분야가 성숙해지면서, 이러한 모델들은 기반 언어 모델링의 방대한 개선 외에도 더 복잡한 능력을 갖추게 되었습니다.
예를 들어 ToolFormer는 "계산기, Q&A 시스템, 두 가지 검색 엔진, 번역 시스템, 달력"을 사용할 수 있었습니다 [@schick2023toolformerlanguagemodelsteach].
얼마 후, Gorilla는 1645개의 API(PyTorch Hub, TensorFlow Hub v2, HuggingFace에서)를 사용하도록 훈련되었으며, 그 평가 도구인 APIBench는 인기 있는 Berkeley Function Calling Leaderboard의 기반이 되었습니다 [@patil2023gorilla].
이러한 초기 모델들 이후, 호출되는 액션의 다양성은 크게 증가했습니다.

도구 사용 모델은 이제 일반적인 언어 모델 상호작용과 깊이 얽혀 있습니다.
모델 컨텍스트 프로토콜 (Model Context Protocol, MCP)은 언어 모델을 외부 데이터 소스(또는 도구)에 연결하는 공통 형식으로 등장했습니다 [@anthropic_mcp_2024].
더 강력한 모델과 더 나은 형식 덕분에, 도구 사용 언어 모델은 Microsoft Office나 Google Workspace 같은 인기 애플리케이션 내 생산성 코파일럿, 과학 분야 [@bran2023chemcrow], 의료 분야 [@li2024mmedagent], Claude Code나 Cursor 같은 코딩 에이전트 [@zhang2024codeagent], 데이터베이스와의 통합, 그리고 여러 자율 워크플로우 등 많은 상황에서 활용됩니다.

도구 사용 모델 평가는 여러 차원을 포함합니다: 도구 이름과 인수 정확도를 위한 정확 일치 지표, 스키마 유효성, 그리고 시뮬레이션 환경에서의 엔드투엔드 작업 완료율.
반복 시도에 걸친 신뢰성도 중요합니다 -- $\tau$-bench는 에이전트가 가끔이 아닌 일관되게 성공하는지를 측정하기 위해 pass^k 지표(pass@k와 구별)를 도입했습니다 [@yao2024taubench].
ToolLLM과 그 ToolBench 데이터셋은 16,000개 이상의 실세계 API에 걸친 도구 사용 훈련 및 평가를 위한 대규모 프레임워크를 제공하며 [@qin2023toollm], Berkeley Function Calling Leaderboard(BFCL)는 함수 호출 정확도에 대한 모델 비교를 위한 인기 있는 벤치마크 (benchmark)로 남아 있습니다 [@patil2023gorilla].

## 생성 중 도구 호출 삽입

함수 호출을 위한 훈련 데이터는 다른 후처리 학습 데이터와 매우 유사하지만, 한 가지 추가 사항이 있습니다: 모델에게 사용 가능한 도구를 알려주는 시스템 프롬프트입니다.
시스템 프롬프트와 JSON 형식으로 제공되는 사용 가능한 도구가 포함된 예시 형식의 데이터 포인트는 아래와 같습니다:
```xml
<system>
You are a function-calling AI model. You are provided with function signatures within <functions></functions> XML tags. You may call one or more functions to assist with the user query. Don't make assumptions about what values to plug into functions.
</system>

<functions>
[
  {
    "name": "search_movies",
    "description": "Search for movies by title and return matching results with IDs.",
    "parameters": {
      "type": "object",
      "properties": {
        "query": {
          "type": "string",
          "description": "The search string for the movie title."
        }
      },
      "required": ["query"]
    }
  },
  {
    "name": "get_movie_details",
    "description": "Fetch detailed information about a movie including cast, runtime, and synopsis.",
    "parameters": {
      "type": "object",
      "properties": {
        "movie_id": {
          "type": "string",
          "description": "The unique identifier for the movie."
        }
      },
      "required": ["movie_id"]
    }
  },
  {
    "name": "get_showtimes",
    "description": "Get movie showtimes for a given location and date.",
    "parameters": {
      "type": "object",
      "properties": {
        "movie_id": {
          "type": "string",
          "description": "The unique identifier for the movie."
        },
        "zip_code": {
          "type": "string",
          "description": "ZIP code for theater location."
        },
        "date": {
          "type": "string",
          "description": "Date for showtimes in YYYY-MM-DD format."
        }
      },
      "required": ["movie_id", "zip_code"]
    }
  }
]
</functions>

<user>
...
</user>
``` 
언어 모델이 생성하는 동안, 위 예시를 따른다면 Star Wars를 검색하기 위해 `search_movies("Star Wars")` 토큰을 생성할 것입니다.
이것은 종종 특수 형식 토큰 안에 인코딩되며, 시퀀스에 삽입되는 다음 토큰에는 도구 출력이 포함됩니다.
이를 통해 모델은 많은 단순 독립형 모델보다 더 도전적인 작업을 수행하는 방법을 학습할 수 있습니다.

인기 있는 도구 사용 형태 중 하나는 코드 실행으로, 모델이 복잡한 논리나 수학 문제에 대한 정확한 답을 얻을 수 있게 합니다.
예를 들어, 언어 모델 실행 내에서의 코드 실행은 추론 모델의 사고 토큰 중에 발생할 수 있습니다.
함수 호출과 마찬가지로, 실행할 코드를 위한 태그(모델이 생성)와 출력을 위한 별도 태그가 먼저 있습니다.
```text
<|user|>
What is the 50th Fibonacci number? (Use the standard F_0=0, F_1=1 indexing.)</s>
<|assistant|>
<think>
Okay, I will compute the 50-th Fibonacci number with a simple loop, then return the result.

<code>
def fib(n):
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a

fib(50)
</code>

<output>
12586269025
</output>
</think>
<answer>
The 50-th Fibonacci number is 12 586 269 025.
</answer>
```

내부적으로 일어나는 일은 언어 모델이 도구 입력 및 출력을 표준 자기회귀 (autoregressive) 방식으로 생성된 토큰과 교차하는 것입니다.
이것을 가능하게 하는 오케스트레이션 루프는 다음과 같습니다:

```python
messages = [...]
while True:
    response = model(messages, tools=tools)
    if not response.tool_calls:
        return response.text

    for call in response.tool_calls:
        result = execute_tool(call.name, call.args)
        messages.append({"role": "tool", "tool_call_id": call.id, "content": result})
```

![도구 사용은 모델 생성과 외부 실행을 교차합니다: 모델은 도구 호출을 발생시킬 때까지 토큰을 생성하고(주황색), 외부 시스템이 도구를 실행하여 출력을 시퀀스에 주입하면(보라색) 모델이 생성을 계속합니다. 모델은 단일 생성에서 여러 도구 호출을 발생시킬 수 있습니다. 훈련 중에 도구 호출과 출력 토큰은 일반적으로 손실에서 마스킹됩니다.](images/tool_use_generation.png){#fig:tool-use-generation}

도구 사용을 위한 훈련은 모델이 이 다른 토큰 흐름에서 예측 가능하게 동작하도록 하는 것입니다 -- 언제 도구 호출을 발생시킬지, 인수를 올바르게 형식화하는 방법, 그리고 결과를 응답에 통합하는 방법을 아는 것입니다.
오픈 모델은 사용자가 즉시 연결할 수 있는 다양한 도구와 함께 작동하도록 훈련되어야 합니다.

## 다단계 도구 추론

OpenAI의 o3 모델은 다단계 도구 사용이 언어 모델과 통합될 수 있는 방식에서 실질적인 도약을 나타냈습니다.
이 동작은 커뮤니티의 훨씬 이전 연구 트렌드와 관련이 있습니다.
예를 들어, ReAct [@yao2023react]는 액션과 추론이 하나의 모델 생성에 교차될 수 있는 방법을 보여주었습니다:

> 이 논문에서 우리는 대규모 언어 모델 (LLM)을 사용하여 추론 흔적과 작업별 액션을 교차된 방식으로 생성하는 것을 탐구합니다. 이는 두 가지 사이의 더 큰 시너지를 허용합니다: 추론 흔적은 모델이 행동 계획을 유도, 추적, 업데이트하고 예외를 처리하는 데 도움을 주며, 액션은 지식 베이스나 환경과 같은 외부 소스에서 추가 정보를 인터페이스하고 수집할 수 있게 합니다.

도구 사용 능력의 공고화와 추론 모델의 도약으로, 다중 턴 도구 사용은 흥미로운 연구 분야로 성장했습니다 [@wang2025ragenunderstandingselfevolutionllm].

## 모델 컨텍스트 프로토콜 (MCP)

모델 컨텍스트 프로토콜 (Model Context Protocol, MCP)은 언어 모델을 외부 데이터 소스 및 정보 시스템에 연결하기 위한 개방형 표준입니다 [@anthropic_mcp_2024].
데이터 계층에서 MCP는 프리미티브에 대한 검색 및 실행 방법과 함께 JSON-RPC 2.0을 사용합니다.
외부 시스템마다 특정 도구 호출 형식을 요구하는 대신, MCP는 표준화된 프로토콜을 통해 모델이 풍부한 맥락 정보에 접근할 수 있게 합니다.

MCP는 이 장의 도구 사용 내용에 간단한 추가 사항입니다 -- 이것은 애플리케이션이 예측 가능한 JSON 스키마로 언어 모델에 컨텍스트(데이터 + 액션)를 전달하는 방법입니다.
모델이 상호작용하는 MCP 서버에는 핵심 프리미티브가 있습니다: 리소스(읽기 전용 데이터 블롭), 프롬프트(템플릿 메시지/워크플로우), 그리고 도구(모델이 호출할 수 있는 함수).
이를 통해 MCP 아키텍처는 다음과 같이 요약될 수 있습니다:

- MCP 서버는 특정 데이터 소스나 기능을 래핑합니다.
- MCP 클라이언트(예: Claude Desktop, IDE 플러그인)는 하나 이상의 서버를 집계합니다.
- 호스트(예: Claude 또는 ChatGPT 애플리케이션)는 사용자/LLM 인터페이스를 제공합니다; 모델 벤더나 백엔드 도구를 전환하는 것은 중간의 클라이언트만 교체하면 됩니다.

MCP는 도구 사용 모델 개발자가 동일한 인프라를 사용하여 서버나 클라이언트를 다른 모델에 연결할 수 있게 하며, 동시에 모델은 외부 구성 요소를 통합하는 데 사용할 수 있는 예측 가능한 형식을 갖습니다.
이 두 가지가 합쳐져 실제 도메인의 도구 사용 모델을 위한 훨씬 더 예측 가능한 개발 환경을 만들어냅니다.

MCP 서버는 표준화된 JSON 스키마를 통해 클라이언트에 도구를 노출합니다:
```json
{
  "name": "get_weather",
  "description": "Get current weather for a location",
  "inputSchema": {
    "type": "object",
    "properties": {
      "location": {
        "type": "string",
        "description": "City name or coordinates"
      }
    },
    "required": ["location"]
  }
}
```

이 도구를 구현하는 최소한의 Python MCP 서버:
```python
from mcp.server import Server
from mcp.types import Tool, TextContent

server = Server("weather-server")

@server.list_tools()
async def list_tools():
    return [Tool(
        name="get_weather",
        description="Get current weather",
        inputSchema={
            "type": "object",
            "properties": {"location": {"type": "string"}},
            "required": ["location"]
        }
    )]

@server.call_tool()
async def call_tool(name: str, arguments: dict):
    if name == "get_weather":
        weather = fetch_weather(arguments["location"])
        return [TextContent(type="text", text=weather)]
```

## 구현 세부 사항

도구 사용 모델을 구현할 때 여러 가지 형식화 및 마스킹 결정이 있습니다:

- **Python vs. JSON 형식화**: 이 장에서 우리는 도구 사용을 JSON 데이터 구조와 Python 코드 모두로 형식화하는 예시를 포함했습니다. 모델은 하나의 구조를 선택하는 경향이 있으며, 업계의 다양한 제공업체들은 다른 형식을 사용합니다.
- **도구 출력 마스킹**: 도구 사용 모델을 훈련할 때 중요한 세부 사항은 도구 출력의 토큰이 모델의 훈련 손실에서 마스킹된다는 것입니다. 이는 모델이 실제 사용 중에 직접 생성하지 않는 시스템의 출력을 예측하는 방법을 학습하지 않도록 합니다(다른 후처리 학습 단계의 프롬프트 마스킹과 유사).
- **도구 호출을 위한 다중 턴 형식화**: 도구 호출 모델을 구현할 때 데이터 로딩 형식에 더 많은 구조를 추가하는 것이 일반적인 관행입니다. 후처리 학습 데이터셋의 표준 관행은 사용자와 어시스턴트 간에 교대하는 메시지 목록(및 종종 시스템 메시지)입니다. 전체 구조는 도구 사용에 대해 동일하지만, 모델의 턴은 각 도구 호출로 구분된 내용의 하위 섹션으로 분할됩니다. 아래에 예시가 있습니다.

```python
messages = [
{
"content": "You are a function calling AI model. You are provided with function signatures within <functions></functions> XML tags. You may call one or more functions to assist with the user query. Don't make assumptions about what values to plug into functions.",
"function_calls": null,
"functions": "[{\"name\": \"live_giveaways_by_type\", \"description\": \"Retrieve live giveaways from the GamerPower API based on the specified type.\", \"parameters\": {\"type\": {\"description\": \"The type of giveaways to retrieve (e.g., game, loot, beta).\", \"type\": \"str\", \"default\": \"game\"}}}]",
"role": "system"
},
{
"content": "Where can I find live giveaways for beta access and games?",
"function_calls": null,
"functions": null,
"role": "user"
},
{
"content": null,
"function_calls": "live_giveaways_by_type(type='beta')\nlive_giveaways_by_type(type='game')",
"functions": null,
"role": "assistant"
}
]
```

- **토큰화 및 메시지 형식 세부 사항**: OpenAI 메시지 형식의 도구 호출은 종종 채팅 템플릿(모델에 전송되는 메시지 형식을 제어하는 코드)을 통한 토큰화를 거쳐, 구조화된 JSON 표현을 원시 토큰 스트림으로 변환합니다. 이 프로세스는 모델 아키텍처마다 다릅니다 -- 일부는 도구 호출을 구분하기 위해 특수 토큰을 사용하고, 다른 것들은 토큰 스트림 내에서 구조화된 형식을 유지합니다. [채팅 템플릿 플레이그라운드](https://huggingface.co/spaces/huggingfacejs/chat-template-playground?modelId=Qwen/Qwen3-8B)는 다른 모델들이 메시지 형식을 토큰 스트림으로 변환하는 방법을 탐색하는 대화형 환경을 제공합니다.
- **추론 토큰 연속성**: 추론 모델이 등장하면서, 답변 전에 별도의 "추론" 토큰 스트림을 갖고, 루프 내 도구 사용과 함께 처리되는 방식에 대한 다양한 구현이 존재합니다. 일부 모델은 단일 턴 내에서 여러 도구 호출 단계에 걸쳐 추론 토큰을 보존하여 여러 도구 호출 전반에 걸쳐 컨텍스트를 유지합니다. 그러나 이러한 토큰은 일반적으로 서빙 비용을 최소화하기 위해 턴 사이에 지워집니다(항상 그런 것은 아닙니다 -- 이것은 설계 결정입니다).
- **제공업체 간 API 형식화** (2025년 7월 기준): 다른 제공업체들은 개념적으로 유사하지만 기술적으로 다른 형식을 사용합니다. OpenAI는 고유 ID가 있는 `tool_calls` 배열을 사용하고, Anthropic은 `<thinking>` 태그와 함께 상세한 `input_schema` 명세를 사용하며, Gemini는 함수 호출 모드(AUTO/ANY/NONE)를 제공합니다. API를 통해 이러한 모델을 사용할 때, 사용 가능한 도구는 JSON 형식으로 정의되고 모델 응답의 도구 출력은 표준 "생성된 토큰"과 별도의 필드에 저장됩니다. 또 다른 예로, 오픈 소스 vLLM 추론 코드베이스는 여러 도구 호출 모드와 모델별 파서를 지원하는 광범위한 파싱 로직을 구현하여 하위 수준 구현 고려 사항에 대한 통찰을 제공합니다 [@kwon2023efficient].
- **스키마 준수 및 제약 디코딩**: 프로덕션 시스템은 종종 제약 디코딩 또는 "엄격 모드" 옵션을 사용하여 유효한 JSON과 올바른 인수 유형을 강제하여 잘못된 출력으로 인한 재시도를 줄입니다. 일부 폐쇄형 모델 제공업체는 구조화된 JSON 출력을 신뢰할 수 있게 하기 위해 추가적인 후처리 학습을 수행하며, 오픈 모델의 경우 VLLM과 같은 시스템에서 추론 플래그로 처리됩니다.
- **도구 출력 컨텍스트 소비**: 도구 출력은 특히 많은 결과를 반환하는 검색이나 검색 도구의 경우 모델의 컨텍스트 윈도우를 빠르게 소비할 수 있습니다. 시스템은 모델이 계속 생성하는 데 필요한 정보를 보존하면서 컨텍스트를 관리 가능한 수준으로 유지하기 위해 도구 출력을 어떻게 잘라내거나, 요약하거나, 페이지를 나눌지 결정해야 합니다.

이를 후처리 학습과 연결하면: 도구 사용 훈련 데이터는 어디서 오는가, 그리고 어떤 목표가 사용되는가?
사람이 작성한 도구 흔적은 수집 비용이 많이 들기 때문에, 대부분의 현대 도구 사용 코퍼스는 합성적이거나 부트스트랩됩니다 -- Toolformer 스타일의 자기 레이블링 [@schick2023toolformerlanguagemodelsteach] 또는 ToolBench [@qin2023toollm]처럼 대규모 생성.
훈련 목표를 위해, 도구 궤적에 대한 지도 미세조정 (SFT)은 기본 형식화와 도구 선택을 가르칩니다.
이것은 행동을 부트스트랩하며 종종 기술의 기반을 확립하는 데 충분합니다.
궤적에 대한 선호도 최적화(예: 직접 선호도 최적화 (DPO))는 도구를 호출할 때와 직접 답변할 때에 대한 결정을 개선할 수 있습니다.
다단계 도구 사용이 있는 에이전트 작업의 경우, 환경 피드백(작업 성공, 제약 만족)을 통한 강화학습 (RL)이 자연스러운 목표가 됩니다 -- 모델은 도구 증강 액션이 실제로 문제를 해결했는지 여부로부터 학습합니다.
