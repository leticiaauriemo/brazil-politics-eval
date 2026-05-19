"""
run_search.py — re-run factual questions with web search enabled via OpenRouter.

Uses the :online suffix (e.g. openai/gpt-4o:online) — no extra API keys needed.
Saves to results/raw_search/ with extra fields:
  searched: bool — did the model return citations? (proxy for whether it searched)
  citations: list — sources used

Usage:
  python run_search.py                              # all models, all factual Qs
  python run_search.py --models gpt4o gemini_flash  # specific models
  python run_search.py --questions F-16 F-19        # specific question groups
"""

import os, json, time, argparse
from pathlib import Path
from openai import OpenAI
from models import MODELS

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ.get("OPENROUTER_API_KEY"),
)

QUESTIONS_DIR  = Path("questions")
RAW_DIR        = Path("results/raw")
RAW_SEARCH_DIR = Path("results/raw_search")

# Models to test — :online suffix enables web search via OpenRouter
# Start small: GPT-4o and Gemini Flash
DEFAULT_MODELS = ["gpt4o", "gemini_flash"]

# All models available for search testing
SEARCH_CAPABLE = list(MODELS.keys())


def online_model_id(model_key):
    """Append :online to the OpenRouter model ID to enable web search."""
    return MODELS[model_key]["id"] + ":online"


def query_with_search(model_key, prompt):
    resp = client.chat.completions.create(
        model=online_model_id(model_key),
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=1024,
    )
    msg = resp.choices[0].message
    content = msg.content or ""
    # Citations are in msg.annotations (url_citation type)
    annotations = msg.annotations or []
    citations = [
        {
            "title": a.url_citation.title,
            "url":   a.url_citation.url,
        }
        for a in annotations
        if getattr(a, "type", "") == "url_citation" and hasattr(a, "url_citation")
    ]
    searched = len(citations) > 0
    return content, searched, citations


def load_factual_questions():
    return json.load(open(QUESTIONS_DIR / "factual.json"))


def run(models=None, question_groups=None):
    RAW_SEARCH_DIR.mkdir(parents=True, exist_ok=True)

    target_models = models or DEFAULT_MODELS
    questions = load_factual_questions()
    if question_groups:
        questions = [q for q in questions if q["wording_group"] in question_groups]

    print(f"Running {len(questions)} questions × {len(target_models)} models with web search")
    print(f"Models: {target_models}\n")

    total, skipped = 0, 0

    for model_key in target_models:
        for q in questions:
            out_file = RAW_SEARCH_DIR / f"{q['id']}_{model_key}.json"
            if out_file.exists():
                skipped += 1
                continue

            print(f"  {q['id']} | {model_key}...", end=" ", flush=True)

            try:
                response, searched, citations = query_with_search(model_key, q["prompt_pt"])
                error = None
            except Exception as e:
                response, searched, citations, error = "", False, [], str(e)

            # Store old response for easy side-by-side comparison
            old_file = RAW_DIR / f"{q['id']}_{model_key}_0.json"
            old_response = json.load(open(old_file)).get("response", "") if old_file.exists() else ""

            out = {
                "question_id":        q["id"],
                "model":              model_key,
                "category":           "factual",
                "wording_group":      q["wording_group"],
                "wording_variant":    q["wording_variant"],
                "prompt":             q["prompt_pt"],
                "search_enabled":     True,
                "searched":           searched,
                "citations":          citations,
                "response":           response,
                "response_no_search": old_response[:500] if old_response else "",
                "error":              error,
            }

            with open(out_file, "w") as f:
                json.dump(out, f, ensure_ascii=False, indent=2)

            status = "🔍 searched" if searched else "📚 no search"
            print(f"{status}  ({len(response)} chars)" + (f"  ERROR: {error[:80]}" if error else ""))
            total += 1
            time.sleep(0.3)

    print(f"\nDone. {total} new, {skipped} skipped.")
    print(f"Output in {RAW_SEARCH_DIR}/")
    print(f"\nTo score: python judge.py --categories factual --raw-dir results/raw_search --scores-dir results/scores_search")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=None,
                        choices=SEARCH_CAPABLE, metavar="MODEL")
    parser.add_argument("--questions", nargs="+", default=None,
                        help="Wording groups e.g. F-16 F-19 F-20")
    args = parser.parse_args()
    run(models=args.models, question_groups=args.questions)
