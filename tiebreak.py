import os
import json
import argparse
import time
from pathlib import Path
from openai import OpenAI
from models import MODELS
from judge import ask_judge, RUBRIC_FACTUAL, score_file_path, extract_json

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ.get("OPENROUTER_API_KEY"),
)

TIEBREAK_JUDGE = "deepseek"
SCORES_DIR  = Path("results/scores")
QUESTIONS_DIR = Path("questions")


def load_questions_map():
    qmap = {}
    path = QUESTIONS_DIR / "factual.json"
    if path.exists():
        for q in json.load(open(path)):
            qmap[q["id"]] = q
    return qmap


def find_disagreements(scores_dir):
    disagreements = []
    for f in scores_dir.glob("*.json"):
        try:
            d = json.load(open(f))
        except Exception:
            continue
        if d.get("category") != "factual":
            continue
        js = d.get("judge_scores", [])
        binaries = [s["binary"] for s in js if s.get("binary") is not None]
        # Only files with exactly 2 judges that disagree
        if len(binaries) == 2 and len(set(binaries)) > 1:
            # Check tiebreak judge not already present
            existing_judges = [s.get("judge") for s in js]
            if TIEBREAK_JUDGE not in existing_judges:
                disagreements.append((f, d))
    return disagreements


def run_tiebreak(scores_dir=None):
    if scores_dir is None:
        scores_dir = SCORES_DIR
    scores_dir = Path(scores_dir)
    qmap = load_questions_map()

    disagreements = find_disagreements(scores_dir)
    print(f"Found {len(disagreements)} files with 2-judge disagreement needing tiebreak")

    total, skipped = 0, 0
    for f, d in disagreements:
        q = qmap.get(d["question_id"])
        if not q:
            continue

        rubric_prompt = RUBRIC_FACTUAL.format(
            ground_truth=q.get("ground_truth", ""),
            resolution=q.get("resolution", ""),
            response=d.get("response", d.get("judge_scores", [{}])[0].get("reasoning", "")),
        )

        # Get response from raw file
        raw_dir = scores_dir.parent / ("raw" if "t10" not in scores_dir.name else "raw_t10")
        raw_file = raw_dir / f"{d['question_id']}_{d['model']}_{d.get('run', 0)}.json"
        if not raw_file.exists():
            continue
        raw = json.load(open(raw_file))

        rubric_prompt = RUBRIC_FACTUAL.format(
            ground_truth=q.get("ground_truth", ""),
            resolution=q.get("resolution", ""),
            response=raw.get("response", ""),
        )

        print(f"  {d['question_id']} {d['model']} run{d.get('run',0)}...", end=" ", flush=True)
        s = ask_judge(TIEBREAK_JUDGE, rubric_prompt)
        time.sleep(0.3)

        # Append tiebreak judge to existing judge_scores
        d["judge_scores"].append({"judge": TIEBREAK_JUDGE, **s})

        # Recompute aggregates
        js = d["judge_scores"]
        valid_scores   = [x["score"]  for x in js if x.get("score")  is not None]
        valid_binaries = [x["binary"] for x in js if x.get("binary") is not None]
        d["score_avg"]   = round(sum(valid_scores)   / len(valid_scores),  2) if valid_scores  else None
        d["binary_avg"]  = round(sum(valid_binaries) / len(valid_binaries), 2) if valid_binaries else None
        d["agreement"]   = (max(valid_scores) - min(valid_scores)) <= 1 if len(valid_scores) >= 2 else None

        with open(f, "w") as fp:
            json.dump(d, fp, ensure_ascii=False, indent=2)

        binaries_str = "/".join(str(x.get("binary", "?")) for x in js)
        print(f"tiebreak={s.get('binary')} → final binary_avg={d['binary_avg']} ({binaries_str})")
        total += 1

    print(f"\nDone. {total} tiebreaks added, {skipped} skipped.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Add a tiebreak judge to 2-judge disagreements")
    parser.add_argument("--scores-dir", default="results/scores",
                        help="Scores directory to scan (default: results/scores)")
    args = parser.parse_args()
    run_tiebreak(args.scores_dir)
