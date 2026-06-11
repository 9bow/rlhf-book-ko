#!/usr/bin/env python3
"""Generate llms.txt and llms-full.txt for the RLHF Book static site."""

from __future__ import annotations

import argparse
import ast
import json
import re
from dataclasses import dataclass
from pathlib import Path


SITE_URL = "https://9bow.github.io/rlhf-book-ko"
PUBLIC_EXCLUDES = {"README.md", "appendix-00-references.md"}


@dataclass(frozen=True)
class Chapter:
    path: Path
    source: str
    title: str
    description: str
    url: str
    body: str


def strip_wrapping_quotes(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        try:
            parsed = ast.literal_eval(value)
        except (SyntaxError, ValueError):
            return value[1:-1]
        if isinstance(parsed, str):
            return parsed
    return value


def quoted_frontmatter_value(value: str) -> str:
    return json.dumps(value, ensure_ascii=False)


def parse_frontmatter(text: str) -> tuple[dict[str, str], str]:
    text = re.sub(r"\A<!--.*?-->\s*", "", text, flags=re.S)
    if not text.startswith("---\n"):
        return {}, text

    _, frontmatter, body = text.split("---", 2)
    meta = {}
    for line in frontmatter.splitlines():
        if ":" not in line or line.startswith((" ", "\t", "-")):
            continue
        key, value = line.split(":", 1)
        meta[key.strip()] = strip_wrapping_quotes(value)
    return meta, body.lstrip()


def public_chapters(root: Path) -> list[Chapter]:
    chapters = []
    for path in sorted((root / "book" / "chapters").glob("*.md")):
        if path.name in PUBLIC_EXCLUDES:
            continue
        meta, body = parse_frontmatter(path.read_text(encoding="utf-8"))
        title = meta.get("page-title") or meta.get("title") or path.stem
        description = meta.get("meta-description", "")
        chapters.append(
            Chapter(
                path=path,
                source=path.relative_to(root).as_posix(),
                title=title,
                description=description,
                url=f"{SITE_URL}/c/{path.stem}",
                body=body.strip(),
            )
        )
    return chapters


def llms_txt(chapters: list[Chapter]) -> str:
    chapter_links = "\n".join(
        f"- [{chapter.title}]({chapter.url}): {chapter.description}"
        for chapter in chapters
    )

    return f"""# RLHF Book 한국어판

> 인간 피드백 기반 강화학습(RLHF), 보상 모델, 선호도 조정, RLVR, 언어 모델 사후 학습을 다루는 무료 온라인 책과 강좌입니다.

Nathan Lambert의 RLHF Book은 지시 미세조정과 선호도 데이터부터 보상 모델링, 정책 최적화, 직접 정렬, 추론 훈련, 평가, 제품 동작까지 현대 언어 모델이 어떻게 사후 학습되는지를 설명합니다.

정식 웹 페이지는 아래 챕터 링크를 사용하세요. `llms-full.txt`는 공개 챕터 소스에서 생성한 단일 Markdown 컨텍스트 파일입니다.

## 핵심 자료

- [전체 텍스트 LLM 컨텍스트]({SITE_URL}/llms-full.txt): 참고문헌 빌드 유틸리티 페이지를 제외한 공개 챕터 Markdown을 이어 붙인 파일입니다.
- [책 홈페이지]({SITE_URL}/): RLHF Book 한국어판의 정식 웹 버전입니다.
- [강좌]({SITE_URL}/course): RLHF와 사후 학습에 관한 무료 강의 및 강좌 자료입니다.
- [모델 라이브러리]({SITE_URL}/library): 지도 미세조정, RLHF, DPO 및 관련 사후 학습 단계의 모델 완성을 비교합니다.
- [RL 치트시트]({SITE_URL}/rl-cheatsheet): PPO, GRPO, RLOO, REINFORCE, DPO 및 관련 RLHF 방법을 한 페이지로 정리한 참고 자료입니다.

## 챕터

{chapter_links}

## 기타

- [PDF]({SITE_URL}/book.pdf): 인쇄 가능한 책 빌드입니다.
- [EPUB]({SITE_URL}/book.epub): 전자책 빌드입니다.
- [한국어판 GitHub 소스](https://github.com/9bow/rlhf-book-ko): 번역본 소스 저장소입니다.
- [원문 GitHub 소스](https://github.com/natolambert/rlhf-book): 책과 강좌의 원문 소스 저장소입니다.
- [ArXiv 논문](https://arxiv.org/abs/2504.12501): RLHF Book의 논문 버전입니다.
- [Manning 도서 페이지](https://www.manning.com/books/the-rlhf-book): The RLHF Book의 출판사 페이지입니다.
"""


def llms_full_txt(chapters: list[Chapter]) -> str:
    parts = [
        "# RLHF Book 한국어판 전체 텍스트",
        "",
        "> 공개 RLHF Book 한국어판 챕터의 Markdown 소스를 이어 붙인 파일입니다.",
        "",
        f"정식 사이트: {SITE_URL}/",
        f"챕터 수: {len(chapters)}",
        "",
        "참고문헌 빌드 유틸리티 페이지는 제외되어 있습니다. 인용은 책 소스의 citation key를 사용합니다.",
    ]

    for chapter in chapters:
        parts.extend(
            [
                "",
                "---",
                f"title: {quoted_frontmatter_value(chapter.title)}",
                f"url: {quoted_frontmatter_value(chapter.url)}",
                f"source: {quoted_frontmatter_value(chapter.source)}",
                "---",
                "",
                chapter.body,
            ]
        )
    return "\n".join(parts).rstrip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory for generated files",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    chapters = public_chapters(root)
    (output_dir / "llms.txt").write_text(llms_txt(chapters), encoding="utf-8")
    (output_dir / "llms-full.txt").write_text(
        llms_full_txt(chapters),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
