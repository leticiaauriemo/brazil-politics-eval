import os
import json
import argparse
import time
from pathlib import Path
from openai import OpenAI
from models import MODELS, CORE_MODELS, EXTENDED_MODELS

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

def query_model(model_key, prompt, temperature=0.0):
    try:
        resp = client.chat.completions.create(
            model=MODELS[model_key]["id"],
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=1024,
        )
        msg = resp.choices[0].message
        content = msg.content or "ERROR: empty response"
        reasoning = getattr(msg, "reasoning_content", None) or None
        citations = getattr(resp, "citations", None) or None
        return content, reasoning, citations
    except Exception as e:
        return f"ERROR: {e}", None, None

def run(categories=None, models=None, language="pt", runs=1, temperature=0.0):
    # Use a subdirectory for non-zero temperature runs to keep raw/ clean
    if temperature != 0.0:
        tag = f"t{int(temperature * 10):02d}"
        out_dir = Path(f"results/raw_{tag}")
    else:
        out_dir = RESULTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    if models and len(models) == 1 and models[0] == "core":
        target_models = CORE_MODELS
    elif models and len(models) == 1 and models[0] == "extended":
        target_models = EXTENDED_MODELS
    else:
        target_models = models or list(MODELS.keys())
    questions = load_questions(categories)

    temp_label = f"temp={temperature}" if temperature != 0.0 else "temp=0"
    print(f"Running {len(questions)} questions × {len(target_models)} models × {runs} run(s) [{temp_label}]")

    for q in questions:
        prompt = q["prompt_pt"] if language == "pt" else q["prompt_en"]
        for model_key in target_models:
            for run_idx in range(runs):
                out_file = out_dir / f"{q['id']}_{model_key}_{run_idx}.json"
                if out_file.exists():
                    print(f"  SKIP {q['id']} | {model_key} | run {run_idx} (exists)")
                    continue

                print(f"  {q['id']} | {model_key} | run {run_idx}...", end=" ", flush=True)
                response, reasoning, citations = query_model(model_key, prompt, temperature)

                result = {
                    "question_id": q["id"],
                    "category": q["category"],
                    "model": model_key,
                    "run": run_idx,
                    "temperature": temperature,
                    "language": language,
                    "prompt": prompt,
                    "response": response,
                }
                if reasoning:
                    result["reasoning"] = reasoning
                if citations:
                    result["citations"] = citations
                for field in ["base_id", "candidate", "pair_id", "subcategory", "level", "mechanism", "target_candidate", "wording_group", "wording_variant"]:
                    if field in q:
                        result[field] = q[field]

                with open(out_file, "w") as f:
                    json.dump(result, f, ensure_ascii=False, indent=2)

                flags = ""
                if reasoning: flags += " [+reasoning]"
                if citations: flags += f" [+{len(citations)} citations]"
                print(f"done ({len(response)} chars{flags})")
                time.sleep(0.5)

    print(f"\nDone. Results saved to {out_dir}/")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Brazil Politics Eval questions against AI models")
    parser.add_argument("--categories", nargs="+", default=None,
                        choices=["factual", "bias", "whataboutism", "voting", "observability"],
                        help="Categories to run (default: all)")
    parser.add_argument("--models", nargs="+", default=None,
                        help="Models to run: model keys, 'core', or 'extended' (default: all)")
    parser.add_argument("--language", default="pt", choices=["pt", "en"])
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Sampling temperature (default: 0.0). Results saved to results/raw_t<N>/ for non-zero values.")
    args = parser.parse_args()
    run(args.categories, args.models, args.language, args.runs, args.temperature)
