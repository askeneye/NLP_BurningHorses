from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import List, Optional

from acl_anthology import Anthology
from rank_bm25 import BM25Okapi


YEARS = {"2024", "2025"}
TOP_K_BM25 = 30
TOP_K_FINAL = 5

NER_QUERY = """
named entity recognition
NER
low-resource ner
generative ner
synthetic data for ner
data augmentation for named entity recognition
entity tagging
token classification named entities
"""

ASSIGNMENT_CONTEXT = """
Student NLP project constraints:
- Dataset/setup: English EWT Universal NER
- Need a baseline stronger than majority class
- Need a somewhat novel but feasible project
- Should support analysis such as ablation, learning curve, per-class F1, or qualitative error analysis
- Limited semester timeframe
"""


@dataclass
class Paper:
    anthology_id: str
    title: str
    abstract: str
    year: str
    url: str

    def full_text(self) -> str:
        return f"{self.title} {self.abstract}".strip()


def normalize_text(text: Optional[str]) -> str:
    if text is None:
        return ""
    return re.sub(r"\s+", " ", str(text)).strip()


def normalize_for_search(text: str) -> str:
    text = normalize_text(text).lower()
    return f" {text} "


def tokenize(text: str) -> List[str]:
    text = normalize_text(text).lower()
    return re.findall(r"[a-z0-9\-]+", text)


def venue_ok(anthology_id: str) -> bool:
    x = anthology_id.lower()
    return (
        x.startswith("2024.acl")
        or x.startswith("2025.acl")
        or x.startswith("2024.findings-acl")
        or x.startswith("2025.findings-acl")
    )


def year_ok(year: Optional[str]) -> bool:
    return str(year) in YEARS


def to_url(anthology_id: str) -> str:
    return f"https://aclanthology.org/{anthology_id}/"


def maybe_call(x):
    return x() if callable(x) else x


def safe_attr(obj, name, default=None):
    if obj is None:
        return default
    val = getattr(obj, name, default)
    try:
        val = maybe_call(val)
    except TypeError:
        pass
    return default if val is None else val


def is_real_paper(anthology_id: str, title: str) -> bool:
    title_l = title.lower()

    # Remove volume/frontmatter entries like 2024.acl-long.0
    if anthology_id.endswith(".0"):
        return False

    # Remove obvious proceedings/frontmatter
    bad_prefixes = [
        "proceedings of",
        "front matter",
        "preface",
        "table of contents",
        "reviewers",
        "program committee",
        "author index",
    ]
    if any(title_l.startswith(prefix) for prefix in bad_prefixes):
        return False

    return True


def extract_papers() -> List[Paper]:
    anthology = Anthology.from_repo()
    papers_attr = getattr(anthology, "papers", None)
    if papers_attr is None:
        raise RuntimeError("anthology.papers not found")

    paper_iter = papers_attr() if callable(papers_attr) else papers_attr

    out: List[Paper] = []

    for p in paper_iter:
        anthology_id = (
            safe_attr(p, "full_id", None)
            or safe_attr(p, "id", None)
            or safe_attr(p, "anthology_id", None)
            or safe_attr(p, "identifier", None)
        )
        anthology_id = normalize_text(anthology_id)

        title = normalize_text(safe_attr(p, "title", ""))
        abstract = normalize_text(safe_attr(p, "abstract", ""))
        year = normalize_text(safe_attr(p, "year", ""))

        if not anthology_id:
            continue
        if not year_ok(year):
            continue
        if not venue_ok(anthology_id):
            continue
        if not title:
            continue
        if not is_real_paper(anthology_id, title):
            continue

        out.append(
            Paper(
                anthology_id=anthology_id,
                title=title,
                abstract=abstract,
                year=year,
                url=to_url(anthology_id),
            )
        )

    return out


def heuristic_prefilter(papers: List[Paper]) -> List[Paper]:
    """
    Keep papers that look plausibly related to NER before BM25.
    """
    terms = [
        " named entity recognition ",
        " ner ",
        " named entities ",
        " entity recognition ",
        " entity tagging ",
        " low-resource ner ",
        " generative ner ",
        " nested ner ",
        " token classification ",
        " sequence labeling ",
        " span classification ",
        " entity extraction ",
    ]

    kept = []
    for p in papers:
        text = normalize_for_search(p.full_text())
        if any(term in text for term in terms):
            kept.append(p)

    return kept


def bm25_retrieve(papers: List[Paper], query: str, top_k: int = TOP_K_BM25) -> List[Paper]:
    if not papers:
        return []

    corpus_tokens = [tokenize(p.full_text()) for p in papers]
    if not corpus_tokens:
        return []

    bm25 = BM25Okapi(corpus_tokens)
    query_tokens = tokenize(query)
    if not query_tokens:
        return []

    scores = bm25.get_scores(query_tokens)

    ranked = sorted(
        zip(papers, scores),
        key=lambda x: x[1],
        reverse=True,
    )

    # Keep only documents with positive score
    ranked = [(p, s) for p, s in ranked if s > 0]

    return [p for p, _ in ranked[:top_k]]


def build_llm_prompt(candidates: List[Paper]) -> str:
    payload = [
        {
            "anthology_id": p.anthology_id,
            "title": p.title,
            "year": p.year,
            "url": p.url,
            "abstract": p.abstract,
        }
        for p in candidates
    ]

    return f"""
You are selecting recent ACL-family papers for a student NLP project.

{ASSIGNMENT_CONTEXT}

Task:
For each paper:
1. Decide whether it is truly relevant to named entity recognition.
2. Give a 1-2 sentence summary.
3. Propose one feasible student project variant.
4. Identify one main risk.
5. Score project fit from 1 to 5.

Then return the best {TOP_K_FINAL} papers overall.

Output valid JSON with this schema:
{{
  "top_5": [
    {{
      "anthology_id": "...",
      "title": "...",
      "year": "...",
      "url": "...",
      "ner_relevance": true,
      "summary": "...",
      "project_variant": "...",
      "risk": "...",
      "fit_score": 1
    }}
  ]
}}

Papers:
{json.dumps(payload, ensure_ascii=False, indent=2)}
""".strip()


def call_llm(prompt: str) -> dict:
    """
    Replace this with your actual LLM call.
    Example options:
    - OpenAI API
    - Ollama
    - local model

    For now it raises on purpose so the retrieval pipeline can be tested first.
    """
    raise NotImplementedError("Connect your LLM here.")


def main():
    print("Loading ACL Anthology...")
    papers = extract_papers()
    print(f"Loaded {len(papers)} ACL/Findings ACL papers from 2024-2025")

    if not papers:
        print("No papers extracted; stopping before retrieval.")
        return

    print("\nSample extracted papers:")
    for p in papers[:10]:
        print(f"{p.year} | {p.anthology_id} | {p.title}")

    papers_prefiltered = heuristic_prefilter(papers)
    print(f"\nAfter heuristic prefilter: {len(papers_prefiltered)} papers")

    if not papers_prefiltered:
        print("No papers survived heuristic prefilter; stopping before BM25.")
        return

    candidates = bm25_retrieve(papers_prefiltered, NER_QUERY, top_k=TOP_K_BM25)
    print(f"\nTop {len(candidates)} BM25 candidates:\n")

    for i, p in enumerate(candidates, 1):
        print(f"{i:02d}. [{p.year}] {p.title}")
        print(f"    ID:  {p.anthology_id}")
        print(f"    URL: {p.url}")
        if p.abstract:
            print(f"    ABS: {p.abstract[:250]}...")
        else:
            print("    ABS: <missing>")
        print()

    if not candidates:
        print("No BM25 candidates found.")
        return

    print("\n--- LLM PROMPT READY ---\n")
    prompt = build_llm_prompt(candidates)
    print(prompt[:5000])

    # Uncomment after wiring your LLM:
    # result = call_llm(prompt)
    # print("\n--- TOP 5 ---\n")
    # print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()