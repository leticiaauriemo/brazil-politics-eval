import os
import json
import argparse
import time
from pathlib import Path
from openai import OpenAI
from models import MODELS

RESULTS_DIR = Path("results/raw")
QUESTIONS_DIR = Path("questions")

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ.get("OPENROUTER_API_KEY"),
)

def load_questions(categories=None):
    files = {
        "factual":        "factual.json",
        "bias":           "bias.json",
        "whataboutism":   "whataboutism.json",
        "voting":         "voting.json",
        "observability":  "observability.json",
    }
    questions = []
    for cat, fname in files.items():
        if categories and cat not in categories:
            continue
        path = QUESTIONS_DIR / fname
        if not path.exists():
            continue
        with open(path) as f:
            data = json.load(f)
        for q in data:
            if cat == "observability":
                for level in q["levels"]:
                    questions.append({
                        "id": f"{q['id']}-{level['level']}",
                        "base_id": q["id"],
                        "category": cat,
                        "mechanism": q["mechanism"],
                        "level": level["level"],
                        "prompt_pt": level["prompt_pt"],
                        "prompt_en": level["prompt_en"],
                        "resolution": q["resolution"],
                    })
            else:
                questions.append(q)
    return questions

def query_model(model_key, prompt):
    try:
        resp = client.chat.completions.create(
            model=MODELS[model_key]["id"],
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=1024,
        )
        return resp.choices[0].message.content
    except Exception as e:
        return f"ERROR: {e}"

def run(categories=None, models=None, language="pt", runs=1):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    target_models = models or list(MODELS.keys())
    questions = load_questions(categories)

    print(f"Running {len(questions)} questions × {len(target_models)} models × {runs} run(s)")

    for q in questions:
        prompt = q["prompt_pt"] if language == "pt" else q["prompt_en"]
        for model_key in target_models:
            for run_idx in range(runs):
                out_file = RESULTS_DIR / f"{q['id']}_{model_key}_{run_idx}.json"
                if out_file.exists():
                    print(f"  SKIP {q['id']} | {model_key} | run {run_idx} (exists)")
                    continue

                print(f"  {q['id']} | {model_key} | run {run_idx}...", end=" ", flush=True)
                response = query_model(model_key, prompt)

                result = {
                    "question_id": q["id"],
                    "category": q["category"],
                    "model": model_key,
                    "run": run_idx,
                    "language": language,
                    "prompt": prompt,
                    "response": response,
                }
                for field in ["base_id", "candidate", "pair_id", "subcategory", "level", "mechanism", "target_candidate"]:
                    if field in q:
                        result[field] = q[field]

                with open(out_file, "w") as f:
                    json.dump(result, f, ensure_ascii=False, indent=2)

                print(f"done ({len(response)} chars)")
                time.sleep(0.5)

    print("\nDone. Results saved to results/raw/")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Brazil Politics Eval questions against AI models")
    parser.add_argument("--categories", nargs="+", default=None,
                        choices=["factual", "bias", "whataboutism", "voting", "observability"],
                        help="Categories to run (default: all)")
    parser.add_argument("--models", nargs="+", default=None,
                        choices=list(MODELS.keys()),
                        help="Models to run (default: all)")
    parser.add_argument("--language", default="pt", choices=["pt", "en"])
    parser.add_argument("--runs", type=int, default=1)
    args = parser.parse_args()
    run(args.categories, args.models, args.language, args.runs)
