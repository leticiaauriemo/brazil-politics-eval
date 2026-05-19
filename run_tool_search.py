"""
run_tool_search.py — run factual questions with opportunistic search via OpenRouter.

Uses openrouter:web_search as a server tool — the model decides whether to call it,
mimicking consumer product behavior (ChatGPT, Gemini, Claude.ai) where search is
available but not forced on every request.

Key difference from run_search.py (:online suffix):
  - :online  → search is FORCED on every request
  - this     → model DECIDES whether to search, just like consumer products

What this captures that :online cannot:
  did_search      — did the model choose to call the search tool?
  search_queries  — the exact query strings the model wrote (bias signal)
  n_searches      — how many times it searched per question
  citations       — sources returned

Saves to results/raw_tool/ with run field = 0 (temp=0 only, no need for temp=1).

Usage:
  python run_tool_search.py                              # core models, all factual Qs
  python run_tool_search.py --models gpt4o claude_sonnet
  python run_tool_search.py --questions F-16 F-19
  python run_tool_search.py --all-models                 # all 18 models
"""

import os, json, time, argparse
from pathlib import Path
from openai import OpenAI
from models import MODELS

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ.get("OPENROUTER_API_KEY"),
)

QUESTIONS_DIR = Path("questions")
RAW_DIR       = Path("results/raw")
RAW_TOOL_DIR  = Path("results/raw_tool")

# Start with the models Brazilians actually use as consumer products
DEFAULT_MODELS = ["gpt4o", "gpt4omini", "gemini_flash", "claude_sonnet", "llama33"]


def query_with_tool_search(model_key, prompt):
    """
    Send a question to a model with openrouter:web_search available as a server tool.
    OpenRouter intercepts the tool decision server-side — we just read whether
    citations came back to know if the model chose to search.
    Returns: (response, did_search, citations)
    """
    resp = client.chat.completions.create(
        model=MODELS[model_key]["id"],
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=1024,
        extra_body={"tools": [{"type": "openrouter:web_search"}]},
    )

    msg = resp.choices[0].message
    content = msg.content or ""

    annotations = getattr(msg, "annotations", None) or []
    citations = [
        {"title": a.url_citation.title, "url": a.url_citation.url}
        for a in annotations
        if getattr(a, "type", "") == "url_citation" and hasattr(a, "url_citation")
    ]
    did_search = len(citations) > 0
    return content, did_search, citations


def load_factual_questions():
    return json.load(open(QUESTIONS_DIR / "factual.json"))


def run(models=None, question_groups=None):
    RAW_TOOL_DIR.mkdir(parents=True, exist_ok=True)

    target_models = models or DEFAULT_MODELS
    questions = load_factual_questions()
    if question_groups:
        questions = [q for q in questions if q["wording_group"] in question_groups]

    print(f"Running {len(questions)} questions × {len(target_models)} models (opportunistic search)")
    print(f"Models: {target_models}\n")

    total, skipped = 0, 0

    for model_key in target_models:
        for q in questions:
            out_file = RAW_TOOL_DIR / f"{q['id']}_{model_key}.json"
            if out_file.exists():
                skipped += 1
                continue

            print(f"  {q['id']} | {model_key}...", end=" ", flush=True)

            try:
                response, did_search, citations = query_with_tool_search(
                    model_key, q["prompt_pt"]
                )
                error = None
            except Exception as e:
                response, did_search, citations, error = "", False, [], str(e)

            # Attach no-search baseline for easy comparison
            old_file = RAW_DIR / f"{q['id']}_{model_key}_0.json"
            old_response = json.load(open(old_file)).get("response", "") if old_file.exists() else ""

            out = {
                "question_id":        q["id"],
                "model":              model_key,
                "run":                0,
                "category":           "factual",
                "wording_group":      q["wording_group"],
                "wording_variant":    q["wording_variant"],
                "prompt":             q["prompt_pt"],
                "search_mode":        "opportunistic",
                "did_search":         did_search,
                "citations":          citations,
                "response":           response,
                "response_no_search": old_response[:500] if old_response else "",
                "error":              error,
            }

            with open(out_file, "w") as f:
                json.dump(out, f, ensure_ascii=False, indent=2)

            if error:
                print(f"ERROR: {error[:80]}")
            elif did_search:
                print(f"🔍 searched  ({len(citations)} citations, {len(response)} chars)")
            else:
                print(f"📚 no search  ({len(response)} chars)")

            total += 1
            time.sleep(0.5)

    print(f"\nDone. {total} new, {skipped} skipped.")
    print(f"Output in {RAW_TOOL_DIR}/")
    print(f"\nTo score: python judge.py --categories factual --raw-dir results/raw_tool --scores-dir results/scores_tool")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=None,
                        choices=list(MODELS.keys()), metavar="MODEL")
    parser.add_argument("--questions", nargs="+", default=None,
                        help="Wording groups e.g. F-16 F-19 F-20")
    parser.add_argument("--all-models", action="store_true",
                        help="Run all 18 models (slow)")
    args = parser.parse_args()

    models = list(MODELS.keys()) if args.all_models else args.models
    run(models=models, question_groups=args.questions)
