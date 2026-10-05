# RLHF Book 첫 재쇄 조판 변경 사항

2026년 10월 · 원문 인쇄 원고에만 해당 · 줄 번호는 현재 제공된 LaTeX 파일을 기준으로 합니다.

최초 인쇄 원고에서 누적된 변경 사항입니다. 표시된 위치에서 각 항목을 교체합니다. 영어 교체 문구는 원문 조판 지침이므로 그대로 유지합니다.

## 장 1 — `Chapter1.tex`

- L41: `true object` → `true objective`.
- L48: `RLHF and related models` → `RLHF and related methods`.
- L167: `markdown formatting` → `Markdown formatting`.
- L244: `ChatBotArena` → `Chatbot Arena`.
- L295: `MT Bench` → `MT-Bench`.
- L303: `OLMo 3.1` → `Olmo 3.1`.
- L312: `Deep-\linebreak[4]Mind merging with Google) or being started` → `Deep-\linebreak[4]Mind merging with Google Brain or new labs being started)`.
- L321: `better, lower, learning rate` → `better, lower learning rate`.

## 장 2 — `Chapter2.tex`

- L73: `Nvidia's Nemotron` → `NVIDIA's Nemotron`.

## 장 3 — `Chapter3.tex`

- L240: `100,000 pairwise prompts` → `100,000 prompts with pairwise completions`.
- L268: `ChatBotArena` → `Chatbot Arena`.

## 장 4 — `Chapter4.tex`

- L251: `OLMo 3` → `Olmo 3`.

## 장 5 — `Chapter5.tex`

- L308: 두 클래스 분류 헤드라는 설명을 각 토큰에서 스칼라 로짓을 출력하는 작은 헤드라는 설명으로 바꿉니다.
- L318: `r \in {0,1}` → `r \in \{0,1\}` (집합의 중괄호가 보이도록 복원합니다).
- L410: `continues to use` → `continues to be inspired by` (Cobbe 등의 원래 정의에 관한 설명).
- L448: PRM 예측기 정의에서 불필요한 닫는 괄호 `)`를 제거합니다.
- L450: `HuggingFace's TRL` → `Hugging Face's TRL`.

## 장 6 — `Chapter6.tex`

- L71: 반환값 정의: `R_{t+1}`, `R_{t+2}`, `R_{t+k+1}` → `r_t`, `r_{t+1}`, `r_{t+k}`.
- L80: `The return definition can also be estimated as` → `The return can also be written recursively as`.
- L85: 재귀적 반환값: `R_{t+1}` → `r_t`.
- 줄 157, 215, 278, 288, 393, 408, 487: 궤적 기댓값: `\tau \sim \pi_\theta` → `\tau \sim p_\theta` (7곳. 기존 아래 첨자 중괄호는 유지합니다).
- L221: 몬테카를로 샘플링: `\tau_i \sim \pi_\theta` → `\tau_i \sim p_\theta`.
- L332: 기본 정책 그래디언트: `R_t` → `G_t`.
- L332: 기본 정책 그래디언트의 합산 항에서 `G_t`를 `\nabla_\theta` 앞에 놓습니다(식 6.20).
- L1103: 정책 비율 설명: `reference model` → `old policy that generated the batch`.

## 장 7 — `Chapter7.tex`

- 줄 426, 475, 486: `OLMo 3` → `Olmo 3` (4곳. L426에 두 번 등장합니다).
- L450: `Phi 4` → `Phi-4`.
- L470: `GPT-OSS` → `gpt-oss`.

## 장 11 — `Chapter11.tex`

- L57: `ChatBotArena` → `Chatbot Arena`.
- L273: `Nvidia GPUs` → `NVIDIA GPUs`.
- L300: `good or great` → `good and great` (문장에서 등장하는 두 곳 모두).

## 장 12 — `Chapter12.tex`

- L1: `\chapter{Synthetic data}` → `\chapter{Synthetic data \& distillation}`.
- L201: `train a weaker version of itself` → `improve its own performance`.
- L223: `Nvidia's work` → `NVIDIA's work`.
- L248: `CriticLLM` → `CritiqueLLM`.

## 장 14 — `Chapter14.tex`

- L163: `LlamaGuard` → `Llama Guard`.

## 장 15 — `Chapter15.tex`

- L347: 절 제목: `Other types of regularization` → `Other tools to control optimization`.
- L349: `These two examples that follow` → `These examples that follow`.
- L351: 하위 절 제목: `Pretraining gradients` → `Pretraining gradients in RL`.
- L373: 기존 DPO/NLL 본문 위에 누락된 하위 절 제목 `Next-token accuracy in DPO`를 추가합니다.
- L402: 하위 절 제목: `Margin-based regularization` → `Margin-based regularization in reward modeling`.

## 장 16 — `Chapter16.tex`

- L14: `SWE-Bench` → `SWE-bench`.
- L200: 잘못된 FLAN 약어 풀이를 제거합니다.
- L295: `SWE-Bench-Verified` → `SWE-bench Verified`.

## 장 17 — `Chapter17.tex`

- L31: `ChatBotArena` → `Chatbot Arena`.

## `Appendix_B.tex`

- L24: `heavy markdown use` → `heavy Markdown use`.
- L160: `MT Bench` → `MT-Bench`.

## `Brief_Lines.txt`

- L15: `\numberline {12}{Synthetic data}` → `\numberline {12}{Synthetic data \& distillation}`.

## `TOC_Lines.txt`

- L235: `\numberline {12}{Synthetic data}` → `\numberline {12}{Synthetic data \& distillation}`.

## `RLHF_Bib.bib`

- L1033: `title={Chatbot arena: An open platform for evaluating llms by human preference},` → `title={{Chatbot Arena}: An Open Platform for Evaluating {LLMs} by Human Preference},`.
- L1318: `title={Judging llm-as-a-judge with mt-bench and chatbot arena},` → `title={Judging {LLM}-as-a-Judge with {MT-Bench} and {Chatbot Arena}},`.
- L1639: `Kimi k1. 5` → `Kimi k1.5`.
- L1681: `LLM Trainin` → `LLM Training`.

## `References.tex`

- 줄 163, 552, 1404, 1627, 2370, 3053, 3384, 3402: `OLMo 3` → `Olmo 3` (8곳).
- 줄 647, 2918: `chatbot arena` → `Chatbot Arena` (“Judging LLM-as-a-judge…” 항목 두 곳 모두).
- 줄 2032, 3028: `Chatbot arena` → `Chatbot Arena` (“Chatbot arena: An open platform…” 항목 두 곳 모두).
