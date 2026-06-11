local json = pandoc.json
local stringify = pandoc.utils.stringify

local function meta_text(meta, key, default)
  local value = meta[key]
  if value == nil then
    return default
  end

  local text = stringify(value)
  if text == "" then
    return default
  end
  return text
end

local function json_block(value)
  return pandoc.MetaBlocks({ pandoc.RawBlock("html", json.encode(value)) })
end

function Meta(meta)
  local lang = meta_text(meta, "lang", "en-US")
  local canonical_url = meta_text(meta, "canonical-url", "https://9bow.github.io/rlhf-book-ko/")
  local default_site_description =
    "RLHF, 선호도 조정, 보상 모델, RLVR, 언어 모델 사후 학습을 다루는 무료 온라인 책과 강좌입니다."
  local default_chapter_description =
    "인간 피드백 기반 강화학습과 언어 모델 사후 학습을 다루는 RLHF Book 한국어판의 한 챕터입니다."

  local book_description = meta_text(meta, "description", default_site_description)
  local chapter_title = meta_text(meta, "page-title", meta_text(meta, "title", "RLHF Book"))
  local chapter_description = meta_text(meta, "meta-description", default_chapter_description)

  meta["book-jsonld"] = json_block({
    ["@context"] = "https://schema.org",
    ["@type"] = "Book",
    name = "인간 피드백 기반 강화학습",
    alternateName = "RLHF Book 한국어판",
    description = book_description,
    author = {
      ["@type"] = "Person",
      name = "Nathan Lambert",
    },
    url = canonical_url,
    image = "https://rlhfbook.com/assets/rlhf-book-cover.png",
    inLanguage = lang,
    isAccessibleForFree = true,
    keywords = "RLHF, 사후 학습, 언어 모델, 보상 모델, 선호도 조정, DPO, RLVR",
    sameAs = {
      "https://github.com/9bow/rlhf-book-ko",
      "https://github.com/natolambert/rlhf-book",
      "https://arxiv.org/abs/2504.12501",
      "https://www.manning.com/books/the-rlhf-book",
    },
  })

  meta["chapter-jsonld"] = json_block({
    ["@context"] = "https://schema.org",
    ["@type"] = "Chapter",
    headline = chapter_title,
    description = chapter_description,
    url = canonical_url,
    image = "https://rlhfbook.com/assets/rlhf-book-share.png",
    author = {
      ["@type"] = "Person",
      name = "Nathan Lambert",
    },
    isPartOf = {
      ["@type"] = "Book",
      name = "인간 피드백 기반 강화학습",
      url = "https://9bow.github.io/rlhf-book-ko/",
    },
    inLanguage = lang,
  })

  meta["breadcrumb-jsonld"] = json_block({
    ["@context"] = "https://schema.org",
    ["@type"] = "BreadcrumbList",
    itemListElement = {
      {
        ["@type"] = "ListItem",
        position = 1,
        name = "RLHF Book 한국어판",
        item = "https://9bow.github.io/rlhf-book-ko/",
      },
      {
        ["@type"] = "ListItem",
        position = 2,
        name = chapter_title,
        item = canonical_url,
      },
    },
  })

  return meta
end
