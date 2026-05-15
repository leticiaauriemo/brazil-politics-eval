import os
import json
import argparse
import time
from pathlib import Path
from openai import OpenAI
from models import MODELS

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ.get("OPENROUTER_API_KEY"),
)

RAW_DIR      = Path("results/raw")
SCORES_DIR   = Path("results/scores")
FOLLOWUP_DIR = Path("results/followup")

SOURCE_PROMPT = """
A seguinte pergunta foi feita para você:

"{question}"

Você respondeu:

"{response}"

Com base nessa resposta: o que embasou essa afirmação? De onde vem essa informação? Cite as fontes ou o raciocínio que te levou a essa conclusão.
""".strip()


def truncate(text, max_chars=600):
    text = text.strip()
    if len(text) <= max_chars:
        return text
    cut = text[:max_chars]
    for sep in [". ", ".\n", "! ", "? "]:
        idx = cut.rfind(sep)
        if idx > max_chars // 2:
            return cut[:idx + 1].strip()
    return cut.strip() + "..."


def load_answers(target):
    """Load answers filtered by target: 'wrong', 'correct', or 'all'."""
    items = []
    for score_file in SCORES_DIR.glob("*.json"):
        try:
            s = json.load(open(score_file))
        except Exception:
            continue
        if not isinstance(s, dict) or s.get("category") != "factual":
            continue

        b = s.get("binary_avg")
        if b is None:
            continue
        if target == "wrong" and b != 0.0:
            continue
        if target == "correct" and b != 1.0:
            continue

        qid   = s["question_id"]
        model = s["model"]
        run   = s.get("run", 0)
        raw_file = RAW_DIR / f"{qid}_{model}_{run}.json"
        if not raw_file.exists():
            continue
        r = json.load(open(raw_file))

        items.append({
            "question_id": qid,
            "model":       model,
            "run":         run,
            "binary":      b,
            "prompt":      r.get("prompt", ""),
            "response":    r.get("response", ""),
        })
    return items


def run_followups(target="wrong"):
    FOLLOWUP_DIR.mkdir(parents=True, exist_ok=True)

    items = load_answers(target)
    print(f"Found {len(items)} {target} answers to follow up on")

    total, skipped = 0, 0
    for item in items:
        out_file = FOLLOWUP_DIR / f"{item['question_id']}_{item['model']}_{item['run']}.json"
        if out_file.exists():
            try:
                existing = json.load(open(out_file))
                if not existing.get("followup_response", "").startswith("ERROR:"):
                    skipped += 1
                    continue
            except Exception:
                pass  # overwrite corrupted files

        prompt = SOURCE_PROMPT.format(
            question=item["prompt"],
            response=truncate(item["response"]),
        )

        print(f"  {item['question_id']} | {item['model']}...", end=" ", flush=True)
        try:
            resp = client.chat.completions.create(
                model=MODELS[item["model"]]["id"],
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=512,
            )
            followup_response = resp.choices[0].message.content or "ERROR: empty response"
        except Exception as e:
            followup_response = f"ERROR: {e}"

        out = {
            "question_id":       item["question_id"],
            "model":             item["model"],
            "run":               item["run"],
            "binary":            item["binary"],
            "target":            target,
            "original_prompt":   item["prompt"],
            "original_response": item["response"],
            "followup_prompt":   prompt,
            "followup_response": followup_response,
        }
        with open(out_file, "w") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)

        print(f"done ({len(followup_response)} chars)")
        total += 1
        time.sleep(0.5)

    print(f"\nDone. {total} follow-ups saved, {skipped} skipped. Output in {FOLLOWUP_DIR}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", default="wrong", choices=["wrong", "correct", "all"],
                        help="Which answers to follow up on (default: wrong)")
    args = parser.parse_args()
    run_followups(args.target)
