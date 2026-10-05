# 학습 레시피 도표

두 개요도는 3장의 그림 5와 6(`book/images/rlhf-basic.png`, `rlhf-complex.png`)에서 사용한 시각적 표현을 확장합니다. 모델은 단순한 테두리 상자, 학습 작업은 기울임꼴 레이블, 연결은 회색의 열린 화살촉과 둥근 선으로 표시합니다.
상자는 모델 체크포인트를 나타내며, 화살표는 학습 방법을 설명합니다. 모든 연결선은 채워진 상자와 테두리 뒤의 배경 레이어에 그립니다.
MOPD 개요도에서는 단계 레이블을 화살표 아래에 놓습니다. 두 줄로 된 "Domain / SFT"와 "Domain / RL" 레이블은 아래쪽 두 수평 화살표 사이의 중앙에 놓으며, 간격을 넓혀 상자와 겹치지 않게 합니다.

공통 스타일은 `../_shared/styles_training_recipes.tex`입니다. `\figdark=1`을 정의하면 사이트의 슬레이트 계열 색상을 사용합니다. 다크 모드 PNG는 배경이 투명하며, 상자는 슬레이트 색, 텍스트·테두리·화살표는 밝은 색입니다.

## 전문 교사 모델과 MOPD

![전문 교사 모델과 MOPD](../../../book/images/rlhf-mopd.png)

`rlhf_mopd_tikz.tex`는 공통 SFT 체크포인트에서 시작해 분야별 교사 모델에 별도의 SFT와 RL을 수행하고, 다중 교사 온-정책 증류(MOPD)로 하나의 범용 학생 모델에 통합하는 과정을 보여줍니다.
`1`, `2`, …, `N` 행은 임의의 분야를 나타내며, 특정 모델이 발표한 교사 수를 뜻하지 않습니다. 각 교사 행의 SFT 상자 아래에는 글쓰기, 코딩, 수학 같은 예시 분야를 표시합니다.
MOPD는 학생의 롤아웃에 대한 교사의 출력 분포로 지식을 전달합니다. 가중치를 평균하는 방법이 아닙니다. 학생 초기화와 롤아웃·손실 계산의 세부 사항은 이 개요도에서 다루지 않습니다.

이 개요는 [MiMo-V2-Flash 보고서 §4.1](https://arxiv.org/html/2601.02780v1#S4.SS1)과 [Nemotron 3 Ultra 보고서 §3](https://research.nvidia.com/labs/nemotron/files/NVIDIA-Nemotron-3-Ultra-Technical-Report.pdf), [강좌 슬라이드](https://rlhfbook.com/teach/course/conversation-01/#14)를 바탕으로 한 일반적인 학습 패턴입니다. 어느 한 모델의 전체 레시피를 그대로 재구성한 것은 아닙니다.
MiMo의 전문화 RL/SFT 설명은 모든 교사가 동일한 두 단계를 따를 것을 요구하지 않습니다. Nemotron의 학생 RL, 교사별 다양한 학습 경로, 여러 증류 라운드, 기타 마무리 단계는 이 개요에서 생략했습니다.

## 순차 강화학습

![순차 강화학습](../../../book/images/rlhf-sequential-rl.png)

`rlhf_sequential_rl_tikz.tex`는 하나의 모델이 전체 SFT 후 추론 RL, 에이전트 RL, 일반 RL을 순서대로 거치는 모습을 보여줍니다. 제공된 GLM-5 슬라이드와 [GLM-5 기술 보고서 §3](https://arxiv.org/html/2602.15763v1#S3)의 단계 순서를 따릅니다.
참고 자료는 GLM-5이므로 GLM-5.2의 레시피를 주장하는 도표가 아닙니다. 순차 RL 단계만 따로 보여주며, 전체 GLM-5 레시피에서 마지막에 수행하는 온-정책 단계 간 증류(§3.5)는 표시하지 않습니다.

## 빌드와 본문 이미지

저장소 루트에서 실행합니다.

```bash
# Both figures; final PNGs at the book's 400 dpi convention.
make -C diagrams training-recipes TIKZ_DENSITY=400

cp diagrams/generated/png/rlhf_mopd_tikz.png book/images/rlhf-mopd.png
cp diagrams/generated/png/rlhf_mopd_tikz-dark.png book/images/rlhf-mopd-dark.png
cp diagrams/generated/svg/rlhf_mopd_tikz.svg book/images/rlhf-mopd.svg
cp diagrams/generated/png/rlhf_sequential_rl_tikz.png book/images/rlhf-sequential-rl.png
cp diagrams/generated/png/rlhf_sequential_rl_tikz-dark.png book/images/rlhf-sequential-rl-dark.png
cp diagrams/generated/svg/rlhf_sequential_rl_tikz.svg book/images/rlhf-sequential-rl.svg
```

이 타깃에는 LaTeX, ImageMagick, 그리고 `pdf2svg` 또는 Poppler의 `pdftocairo`가 필요합니다. PNG의 기본 해상도는 800 dpi이며, 빠른 미리보기에는 300 dpi를 사용합니다. PDF/SVG 출력은 벡터 형식을 유지합니다.
중간 파일과 생성된 출력은 모두 `diagrams/generated/`에 저장합니다. 검토를 마친 웹·인쇄용 이미지만 `book/images/`에 커밋합니다. 다크 모드 PNG는 어두운 배경 위에서 미리 확인합니다.

MOPD 개요도는 3장의 DeepSeek R1 다음에 오는 "MOPD와 에이전트로의 전환" 절에 삽입하며, 다크 모드 이미지도 제공합니다.
순차 RL 개요도는 별도 이미지로 사용할 수 있습니다. 본문에서는 단계 간 증류를 포함한 전체 레시피를 보여주기 위해 GLM-5의 원본 그림 5를 사용합니다. 추출 방법은 아래에 설명합니다.
두 개요도의 삽입 예시는 다음과 같습니다.

```markdown
![분야별 사후 학습 개요: 공통 SFT, 분야별 SFT와 RL, 다중 교사 온-정책 증류를 통한 하나의 학생 모델로의 통합.](images/rlhf-mopd.png){#fig:rlhf-mopd data-dark-src="images/rlhf-mopd-dark.png"}

![순차 사후 학습 개요: 전체 SFT 후 추론, 에이전트, 일반 강화학습 수행.](images/rlhf-sequential-rl.png){#fig:rlhf-sequential-rl data-dark-src="images/rlhf-sequential-rl-dark.png"}
```

## GLM-5 그림 5 재현

`book/images/glm5-pipeline.png`와 `.svg`는 GLM-5 연구팀의 *GLM-5: from Vibe Coding to Agentic Engineering*, [arXiv:2602.15763v2](https://arxiv.org/abs/2602.15763v2)의 그림 5를 재현합니다. 라이선스는 [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)입니다.
원본 그림과 색상을 유지하며, 사이트의 두 테마에서 모두 흰 배경을 사용합니다. 3장에서 순차 RL만 보여주던 개요도를 대체하며 기존 그림 참조 식별자는 유지합니다.

PDF 4쪽에서 그림 설명과 주변 텍스트를 제외하고 잘라냅니다. 왼쪽 아래를 기준으로 한 PDF 포인트 좌표는 `(107, 496.331, 505.19508, 721.00016)`이며, 그림 주위에 1포인트 여백을 포함합니다.
PNG는 400 dpi(2213 × 1249픽셀)로 렌더링하고 PDF와 SVG는 벡터를 유지합니다. 원본 PDF의 SHA-256은 `e20742ff36e08dc361de6973f7f72ad38e107edf8fd92d2a777a8428fc9b8f0e`입니다.

`pypdf`와 Poppler를 설치한 환경에서 저장소 루트에서 실행합니다.

```bash
curl -L --fail https://arxiv.org/pdf/2602.15763v2 \
  -o diagrams/generated/pdf/glm5-source.pdf
uv run python diagrams/scripts/extract_glm5_pipeline.py
cp diagrams/generated/png/glm5-pipeline.png book/images/
cp diagrams/generated/svg/glm5-pipeline.svg book/images/
```

## Nature 그림 2 재현

`book/images/deepseek-r1-pipeline.png`와 `.svg`는 Guo 등(DeepSeek-AI 연구팀)의 *Nature* **645**, 633–638 (2025), [doi:10.1038/s41586-025-09422-z](https://doi.org/10.1038/s41586-025-09422-z)의 그림 2를 재현합니다.
[출판된 PDF](https://www.nature.com/articles/s41586-025-09422-z.pdf)의 인쇄 637쪽(PDF 5쪽)은 별도의 저작권 표시가 없는 한 논문과 그림에 [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)을 적용한다고 명시합니다. [그림 2](https://www.nature.com/articles/s41586-025-09422-z/figures/2)에는 별도 제한이 없습니다.
재현한 그림의 저작권 표시는 © The Author(s) 2025, CC BY 4.0입니다. 본문 그림 설명에서 논문과 라이선스를 링크하며, 아래에 추출 방법을 기록합니다.

그림 자체는 변경하지 않습니다. 인쇄 635쪽(PDF 3쪽)의 그림 2에 2포인트 여백을 더해 잘라내고, 출판사의 그림 설명과 주변 텍스트는 제외합니다.
왼쪽 아래를 기준으로 한 PDF 포인트 좌표는 `(81.736, 69.398, 514.636, 276.769)`입니다. PNG는 400 dpi에서 2405 × 1153픽셀이며, SVG는 벡터 경로를 유지합니다.
잘라낸 PDF는 `diagrams/generated/pdf/`에 저장합니다. 본문 PNG는 원본의 흰 배경을 유지합니다.

`pypdf`와 Poppler를 설치한 환경에서 저장소 루트에서 실행합니다.

```bash
mkdir -p diagrams/generated/{pdf,png,svg}
curl -L --fail 'https://www.nature.com/articles/s41586-025-09422-z.pdf' \
  -o diagrams/generated/pdf/deepseek-r1-nature-source.pdf
uv run python - <<'PYTHON'
from pathlib import Path
from pypdf import PdfReader, PdfWriter
from pypdf.generic import RectangleObject

reader = PdfReader('diagrams/generated/pdf/deepseek-r1-nature-source.pdf')
page = reader.pages[2]
box = RectangleObject((81.736, 69.398, 514.636, 276.769))
for name in ('mediabox', 'cropbox', 'trimbox', 'bleedbox', 'artbox'):
    setattr(page, name, box)
writer = PdfWriter()
writer.add_page(page)
writer.add_metadata({
    '/Title': 'DeepSeek-R1 multistage pipeline (Nature Figure 2)',
    '/Author': 'Daya Guo et al. (DeepSeek-AI Team)',
    '/Subject': 'Figure 2 from Nature 645, 633–638 (2025), doi:10.1038/s41586-025-09422-z. CC BY 4.0. Cropped to the original figure; artwork unchanged.',
})
with Path('diagrams/generated/pdf/deepseek-r1-pipeline.pdf').open('wb') as output:
    writer.write(output)
PYTHON
pdftocairo -svg diagrams/generated/pdf/deepseek-r1-pipeline.pdf \
  diagrams/generated/svg/deepseek-r1-pipeline.svg
pdftoppm -cropbox -r 400 -png -singlefile \
  diagrams/generated/pdf/deepseek-r1-pipeline.pdf \
  diagrams/generated/png/deepseek-r1-pipeline
cp diagrams/generated/png/deepseek-r1-pipeline.png book/images/
cp diagrams/generated/svg/deepseek-r1-pipeline.svg book/images/
```

추출 당시 원본 PDF의 SHA-256은 `916fdaafc3b44143a9744481d079744f773d73d06372f82b1dba1076525927f2`였습니다.
Nature에서 제공하는 [공식 1787 × 847 PNG](https://media.springernature.com/full/springer-static/image/art%3A10.1038%2Fs41586-025-09422-z/MediaObjects/41586_2025_9422_Fig2_HTML.png)를 이용해 벡터로 잘라낸 결과가 그림 전체를 보존하는지 확인했습니다.
