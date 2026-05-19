import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import pandas as pd
import plotly.express as px
import streamlit as st
from models import MODELS

st.set_page_config(
    page_title="Brazil Politics Eval",
    page_icon="🇧🇷",
    layout="wide",
)

ROOT = Path(__file__).parent.parent
SCORES_DIR      = ROOT / "results" / "scores"
SCORES_T10_DIR  = ROOT / "results" / "scores_t10"
SCORES_TOOL_DIR = ROOT / "results" / "scores_tool"
FOLLOWUP_DIR    = ROOT / "results" / "followup"
RAW_DIR         = ROOT / "results" / "raw"
RAW_TOOL_DIR    = ROOT / "results" / "raw_tool"
RAW_T10_DIR     = ROOT / "results" / "raw_t10"
QUESTIONS_DIR   = ROOT / "questions"
CATEGORIES_FILE      = ROOT / "results" / "categories.json"
DISAGREEMENTS_FILE   = ROOT / "results" / "disagreement_review.json"
GROUP_TYPES_FILE     = ROOT / "results" / "question_group_types.json"

# Models whose tool-search results are invalid (empty responses — use baseline instead)
TOOL_SEARCH_BROKEN = {"perplexity", "llama4"}

MODEL_DISPLAY = {k: v["display"] for k, v in MODELS.items()}

FAILURE_CATEGORIES = [
    ("temporal_stale",      "Temporal — Stale",       "Had info but it was outdated"),
    ("temporal_missing",    "Temporal — Missing",     "Post-cutoff, no knowledge at all"),
    ("semantic_conflation", "Semantic Conflation",    "Confused similar concepts (e.g. anulado vs inocentado)"),
    ("semantic_escape",     "Semantic Escape",        "Used a related but different word to avoid the core claim (e.g. governo vs governador)"),
    ("false_confidence",    "False Confidence",       "Wrong but stated with no hedging"),
    ("hallucination",       "Hallucination",          "Invented specific facts, names, or numbers"),
    ("partial_correct",     "Partially Correct",      "Got the broad fact right, missed critical detail"),
    ("political_framing",   "Political Framing",      "Answer shaped by perceived political lean"),
    ("refusal",             "Refusal",                "Declined to answer a factual question"),
    ("wording_sensitive",   "Wording Sensitive",      "Answer changed across wording variants"),
]


@st.cache_data
def load_group_types():
    if not GROUP_TYPES_FILE.exists():
        return set(), set()
    d = json.load(open(GROUP_TYPES_FILE))
    true_variants = set(d["classification"]["true_wording_variants"]["groups"])
    mostly_same   = set(d["classification"]["mostly_same_fact"]["groups"].keys())
    return true_variants, mostly_same


@st.cache_data(ttl=30)
def load_scores():
    rows = []
    for f in SCORES_DIR.glob("*.json"):
        try:
            d = json.load(open(f))
            if not isinstance(d, dict):
                continue
            rows.append(d)
        except Exception:
            pass
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    if "model" in df.columns:
        df["model_display"] = df["model"].map(MODEL_DISPLAY).fillna(df["model"])
    return df


@st.cache_data(ttl=30)
def load_scores_tool():
    """
    Tool-search scores (opportunistic search — model decides).
    For Perplexity/Grok whose tool responses were empty, falls back to baseline scores.
    """
    rows = []
    # Tool search results for models where it worked
    if SCORES_TOOL_DIR.exists():
        for f in SCORES_TOOL_DIR.glob("*.json"):
            try:
                d = json.load(open(f))
                if not isinstance(d, dict):
                    continue
                if d.get("model") in TOOL_SEARCH_BROKEN:
                    continue  # skip — use baseline for these
                rows.append(d)
            except Exception:
                pass
    # Baseline (native search) for broken models
    for f in SCORES_DIR.glob("*.json"):
        try:
            d = json.load(open(f))
            if not isinstance(d, dict):
                continue
            if d.get("model") not in TOOL_SEARCH_BROKEN:
                continue
            if d.get("category") != "factual":
                continue
            rows.append(d)
        except Exception:
            pass
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    if "model" in df.columns:
        df["model_display"] = df["model"].map(MODEL_DISPLAY).fillna(df["model"])
    return df


@st.cache_data(ttl=30)
def load_followups():
    items = []
    cats = {}
    if CATEGORIES_FILE.exists():
        cats = json.load(open(CATEGORIES_FILE))
    for f in sorted(FOLLOWUP_DIR.glob("*.json")):
        try:
            d = json.load(open(f))
            key = f"{d['question_id']}_{d['model']}_{d['run']}"
            d["_key"] = key
            d["_assigned"] = cats.get(key, {})
            d["_wording_group"] = d["question_id"].rsplit("-", 1)[0]
            items.append(d)
        except Exception:
            pass
    return items


@st.cache_data(ttl=60)
def load_raw(directory):
    raw = {}
    p = Path(directory)
    if not p.exists():
        return raw
    for f in p.glob("*.json"):
        try:
            d = json.load(open(f))
            key = (d["question_id"], d["model"], d.get("run", 0))
            raw[key] = d
        except Exception:
            pass
    return raw


@st.cache_data
def load_questions(category):
    fname = {
        "factual": "factual.json",
        "bias": "bias.json",
        "whataboutism": "whataboutism.json",
        "voting": "voting.json",
        "observability": "observability.json",
    }.get(category, "")
    path = QUESTIONS_DIR / fname
    if not path.exists():
        return []
    return json.load(open(path))


def save_category(key, category, notes=""):
    cats = {}
    if CATEGORIES_FILE.exists():
        cats = json.load(open(CATEGORIES_FILE))
    cats[key] = {"category": category, "notes": notes}
    json.dump(cats, open(CATEGORIES_FILE, "w"), indent=2, ensure_ascii=False)
    st.cache_data.clear()


def binary_chip(val):
    if val is None:
        return "⚪"
    return "✅" if val >= 0.5 else "❌"


def render_judge_scores(score_row):
    """Show all 3 judge scores and their reasoning inline."""
    judge_scores = score_row.get("judge_scores", [])
    if not judge_scores:
        return
    for js in judge_scores:
        judge = MODEL_DISPLAY.get(js.get("judge", ""), js.get("judge", "?"))
        binary = js.get("binary")
        score  = js.get("score")
        reason = js.get("reasoning", "")
        chip   = "✅" if binary == 1 else "❌"
        st.caption(f"{chip} **{judge}** — score {score}/5 — _{reason}_")


# ── FACTUAL OVERVIEW ──────────────────────────────────────────────────────────

def tab_factual_overview(df):
    st.header("Factual Accuracy")

    condition = st.radio(
        "Condition",
        ["Tool Search (consumer behavior)", "No Search (training only)"],
        index=0,
        horizontal=True,
        help="Tool Search: model decides when to search, like ChatGPT/Gemini/Claude.ai. "
             "Perplexity and Grok shown using their baseline (native search always on).",
    )

    if condition.startswith("Tool"):
        df = load_scores_tool()
        st.caption("Model decides whether to search — closest to consumer product behavior. "
                   "Perplexity & Grok use their no-search baseline (their tool search was incompatible).")
    else:
        st.caption("Binary pass/fail: did the model state the core fact correctly from training data alone?")

    if df.empty or "category" not in df.columns:
        st.info("No scores yet.")
        return

    fdf = df[df["category"] == "factual"].copy()
    if fdf.empty:
        st.info("No factual scores found.")
        return

    # Aggregate: for each (model, wording_group) average binary_avg across runs & variants
    fdf["wording_group"] = fdf.get("wording_group", fdf["question_id"].str.rsplit("-", n=1).str[0])
    agg = (
        fdf.groupby(["model", "model_display", "wording_group"])["binary_avg"]
        .mean()
        .reset_index()
    )

    # Model ranking
    model_rank = (
        agg.groupby(["model", "model_display"])["binary_avg"]
        .mean()
        .reset_index()
        .sort_values("binary_avg", ascending=False)
    )
    model_rank["pct"] = (model_rank["binary_avg"] * 100).round(1)

    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader("Model Ranking")
        fig = px.bar(
            model_rank, y="model_display", x="pct", orientation="h",
            color="pct", color_continuous_scale="RdYlGn", range_color=[0, 100],
            text=model_rank["pct"].astype(str) + "%",
            labels={"model_display": "", "pct": "Accuracy %"},
        )
        fig.update_traces(textposition="outside")
        fig.update_layout(coloraxis_showscale=False, height=500, yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Accuracy Heatmap — Model × Question")
        pivot = agg.pivot(index="model_display", columns="wording_group", values="binary_avg")
        # order rows by overall accuracy
        pivot = pivot.loc[model_rank["model_display"].values]
        # order cols by difficulty (hardest first)
        col_order = pivot.mean(axis=0).sort_values().index
        pivot = pivot[col_order]

        fig2 = px.imshow(
            pivot, color_continuous_scale="RdYlGn", zmin=0, zmax=1,
            text_auto=".0%", aspect="auto",
            labels={"color": "Accuracy"},
        )
        fig2.update_layout(height=500)
        st.plotly_chart(fig2, use_container_width=True)

    # Wording sensitivity — only true wording variant groups
    st.subheader("Wording Sensitivity")
    true_variants, mostly_same = load_group_types()
    valid_for_wording = true_variants | mostly_same
    st.caption(
        f"How much does accuracy vary across A/B/C phrasings of the **same** question? "
        f"Only the {len(true_variants)} true-variant groups are included "
        f"(+ {len(mostly_same)} mostly-same groups). "
        f"The 11 groups with different questions about the same topic are excluded."
    )

    wording_fdf = fdf[fdf["wording_group"].isin(valid_for_wording)] if valid_for_wording else fdf
    variant_agg = (
        wording_fdf.groupby(["model_display", "wording_group", "wording_variant"])["binary_avg"]
        .mean()
        .reset_index()
    )
    sensitivity = (
        variant_agg.groupby(["model_display", "wording_group"])["binary_avg"]
        .std()
        .reset_index()
        .rename(columns={"binary_avg": "std"})
    )
    sensitivity["std"] = sensitivity["std"].fillna(0)
    top_sensitive = sensitivity.groupby("wording_group")["std"].mean().sort_values(ascending=False).head(10)

    fig3 = px.bar(
        top_sensitive.reset_index(),
        x="wording_group", y="std",
        title="Top 10 True-Variant Groups with Highest Wording Sensitivity (std dev of accuracy across A/B/C)",
        labels={"wording_group": "Question Group", "std": "Std Dev"},
        color="std", color_continuous_scale="Reds",
    )
    fig3.update_layout(coloraxis_showscale=False)
    st.plotly_chart(fig3, use_container_width=True)

    # Hardest questions
    st.subheader("Hardest Questions")
    hard = (
        agg.groupby("wording_group")["binary_avg"]
        .mean()
        .reset_index()
        .sort_values("binary_avg")
        .head(10)
    )
    hard["pct"] = (hard["binary_avg"] * 100).round(1)
    st.dataframe(
        hard[["wording_group", "pct"]].rename(columns={"wording_group": "Question", "pct": "Accuracy %"}),
        use_container_width=True, hide_index=True,
    )


# ── FOLLOW-UP EXPLORER ────────────────────────────────────────────────────────

def tab_followups():
    st.header("Follow-up Explorer")
    st.caption("Browse wrong answers and assign failure categories. 420 total wrong answers.")

    items = load_followups()
    if not items:
        st.info("No follow-up files found. Run `python followup.py` first.")
        return

    cats = {}
    if CATEGORIES_FILE.exists():
        cats = json.load(open(CATEGORIES_FILE))

    categorized = sum(1 for i in items if i["_assigned"])
    st.progress(categorized / len(items), text=f"{categorized} / {len(items)} categorized")

    # Filters
    col1, col2, col3 = st.columns(3)
    with col1:
        models_available = sorted(set(i["model"] for i in items))
        model_filter = st.selectbox("Model", ["All"] + models_available, key="fu_model")
    with col2:
        groups_available = sorted(set(i["_wording_group"] for i in items))
        group_filter = st.selectbox("Question", ["All"] + groups_available, key="fu_question")
    with col3:
        status_filter = st.selectbox("Status", ["All", "Uncategorized", "Categorized"], key="fu_status")

    filtered = items
    if model_filter != "All":
        filtered = [i for i in filtered if i["model"] == model_filter]
    if group_filter != "All":
        filtered = [i for i in filtered if i["_wording_group"] == group_filter]
    if status_filter == "Uncategorized":
        filtered = [i for i in filtered if not i["_assigned"]]
    elif status_filter == "Categorized":
        filtered = [i for i in filtered if i["_assigned"]]

    st.caption(f"Showing {len(filtered)} items")

    for item in filtered:
        assigned = item["_assigned"]
        assigned_label = assigned.get("category", "") if assigned else ""
        chip = f"✅ {assigned_label}" if assigned_label else "⬜ uncategorized"

        with st.expander(
            f"{chip}  |  **{item['question_id']}** — {MODEL_DISPLAY.get(item['model'], item['model'])}",
            expanded=not bool(assigned_label),
        ):
            col_orig, col_followup = st.columns(2)

            with col_orig:
                st.markdown("**Original question**")
                st.info(item.get("original_prompt", ""))
                st.markdown("**Model's wrong answer**")
                st.error(item.get("original_response", "")[:800])

            with col_followup:
                st.markdown("**Follow-up: what was your source?**")
                st.success(item.get("followup_response", "")[:800])

            st.divider()
            col_cat, col_notes, col_save = st.columns([2, 3, 1])
            with col_cat:
                cat_options = [""] + [c[0] for c in FAILURE_CATEGORIES]
                cat_labels  = ["— select category —"] + [f"{c[0]}  ({c[2]})" for c in FAILURE_CATEGORIES]
                current_idx = cat_options.index(assigned_label) if assigned_label in cat_options else 0
                selected = st.selectbox(
                    "Failure category",
                    options=cat_options,
                    format_func=lambda x: cat_labels[cat_options.index(x)],
                    index=current_idx,
                    key=f"cat_{item['_key']}",
                )
            with col_notes:
                notes = st.text_input(
                    "Notes (optional)",
                    value=assigned.get("notes", "") if assigned else "",
                    key=f"notes_{item['_key']}",
                )
            with col_save:
                st.markdown("&nbsp;", unsafe_allow_html=True)
                if st.button("Save", key=f"save_{item['_key']}", disabled=not selected):
                    save_category(item["_key"], selected, notes)
                    st.success("Saved!")
                    st.rerun()


# ── JUDGE DISAGREEMENTS ───────────────────────────────────────────────────────

def load_disagreement_review():
    if DISAGREEMENTS_FILE.exists():
        return json.load(open(DISAGREEMENTS_FILE))
    return {}

def save_disagreement_review(key, verdict, notes=""):
    reviews = load_disagreement_review()
    reviews[key] = {"verdict": verdict, "notes": notes}
    json.dump(reviews, open(DISAGREEMENTS_FILE, "w"), indent=2, ensure_ascii=False)
    st.cache_data.clear()

@st.cache_data(ttl=30)
def load_binary_splits():
    splits = []
    for f in SCORES_DIR.glob("*.json"):
        try:
            d = json.load(open(f))
        except: continue
        if not isinstance(d, dict) or d.get("category") != "factual": continue
        js = d.get("judge_scores", [])
        binaries = [s.get("binary") for s in js if s.get("binary") is not None]
        if len(binaries) >= 2 and len(set(binaries)) > 1:
            scores = [s.get("score") for s in js if s.get("score") is not None]
            spread = max(scores) - min(scores) if scores else 0
            splits.append({**d, "_spread": spread})
    return sorted(splits, key=lambda d: -d["_spread"])

def tab_disagreements():
    st.header("Judge Disagreements")
    st.caption(
        "167 cases where judges split on binary pass/fail. "
        "Mark each as: **keep** (judge error), **ambiguous** (question needs revision), or **remove**."
    )

    splits = load_binary_splits()
    reviews = load_disagreement_review()
    questions_map = {q["id"]: q for q in load_questions("factual")}
    raw = load_raw(str(RAW_DIR))

    reviewed   = sum(1 for s in splits if f"{s['question_id']}_{s['model']}_{s.get('run',0)}" in reviews)
    st.progress(reviewed / len(splits), text=f"{reviewed} / {len(splits)} reviewed")

    # Filter
    col1, col2 = st.columns(2)
    with col1:
        verdict_filter = st.selectbox("Show", ["All", "Unreviewed", "keep", "ambiguous", "remove"], key="dis_verdict")
    with col2:
        qg_filter = st.selectbox("Question group", ["All"] + sorted(set(
            s.get("wording_group", s["question_id"].rsplit("-",1)[0]) for s in splits
        )), key="dis_qgroup")

    filtered = splits
    if qg_filter != "All":
        filtered = [s for s in filtered if s.get("wording_group", s["question_id"].rsplit("-",1)[0]) == qg_filter]
    if verdict_filter == "Unreviewed":
        filtered = [s for s in filtered if f"{s['question_id']}_{s['model']}_{s.get('run',0)}" not in reviews]
    elif verdict_filter in ("keep", "ambiguous", "remove"):
        filtered = [s for s in filtered if reviews.get(f"{s['question_id']}_{s['model']}_{s.get('run',0)}", {}).get("verdict") == verdict_filter]

    st.caption(f"Showing {len(filtered)} items")

    for d in filtered:
        key = f"{d['question_id']}_{d['model']}_{d.get('run',0)}"
        existing = reviews.get(key, {})
        verdict = existing.get("verdict", "")
        verdict_chip = {"keep": "✅", "ambiguous": "⚠️", "remove": "🗑️"}.get(verdict, "⬜")

        js = d.get("judge_scores", [])
        scores_str  = " / ".join(str(s.get("score","?")) for s in js)
        binary_str  = " / ".join(str(s.get("binary","?")) for s in js)

        wg = d.get("wording_group", d["question_id"].rsplit("-",1)[0])
        true_v, mostly_s = load_group_types()
        badge = "✓ wording variant" if wg in true_v else ("~ mostly same" if wg in mostly_s else "✗ topic cluster")
        with st.expander(
            f"{verdict_chip} **{d['question_id']}** [{badge}] — {MODEL_DISPLAY.get(d['model'], d['model'])}  "
            f"| scores: {scores_str}  | binary: {binary_str}",
            expanded=not bool(verdict),
        ):
            # Question ground truth
            q = questions_map.get(d["question_id"])
            if q:
                st.info(f"**{q['prompt_pt']}**  \n_Ground truth: {q['ground_truth']}_")

            # Model's response
            r = raw.get((d["question_id"], d["model"], d.get("run", 0)))
            if r:
                st.markdown("**Model response:**")
                st.write((r.get("response") or "*no response recorded*")[:600])

            st.divider()

            # All 3 judge scores with reasoning
            cols = st.columns(len(js))
            for col, js_item in zip(cols, js):
                with col:
                    b = js_item.get("binary")
                    chip = "✅" if b == 1 else "❌"
                    judge_name = MODEL_DISPLAY.get(js_item.get("judge",""), js_item.get("judge","?"))
                    st.markdown(f"{chip} **{judge_name}**  \nscore: {js_item.get('score','?')}/5")
                    st.caption(js_item.get("reasoning", ""))

            st.divider()

            # Verdict
            v_col, n_col, s_col = st.columns([2, 3, 1])
            with v_col:
                options   = ["", "keep", "ambiguous", "remove"]
                labels    = ["— decide —", "✅ Keep (judge error)", "⚠️ Ambiguous (question unclear)", "🗑️ Remove from analysis"]
                cur_idx   = options.index(verdict) if verdict in options else 0
                selected  = st.selectbox("Verdict", options, format_func=lambda x: labels[options.index(x)],
                                         index=cur_idx, key=f"v_{key}")
            with n_col:
                notes = st.text_input("Notes", value=existing.get("notes",""), key=f"n_{key}")
            with s_col:
                st.markdown("&nbsp;", unsafe_allow_html=True)
                if st.button("Save", key=f"s_{key}", disabled=not selected):
                    save_disagreement_review(key, selected, notes)
                    st.success("Saved!")
                    st.rerun()


# ── WORDING COMPARISON ────────────────────────────────────────────────────────

@st.cache_data(ttl=30)
def load_run_consistency():
    from collections import defaultdict
    by_qm = defaultdict(list)
    for f in SCORES_DIR.glob("*.json"):
        try:
            d = json.load(open(f))
        except: continue
        if not isinstance(d, dict) or d.get("category") != "factual": continue
        b = d.get("binary_avg")
        if b is None: continue
        key = (d["question_id"], d["model"],
               d.get("wording_group", d["question_id"].rsplit("-",1)[0]),
               d.get("wording_variant",""))
        by_qm[key].append((d.get("run", 0), b))

    rows = []
    for (qid, model, wg, wv), runs in by_qm.items():
        if len(runs) < 2: continue
        runs_sorted = sorted(runs, key=lambda x: x[0])
        binaries = [b for _, b in runs_sorted]
        binary_labels = [b >= 0.5 for b in binaries]
        pattern = "".join("✅" if b else "❌" for b in binary_labels)
        flips = len(set(binary_labels)) > 1
        rows.append({
            "question_id": qid,
            "wording_group": wg,
            "wording_variant": wv,
            "model": model,
            "model_display": MODEL_DISPLAY.get(model, model),
            "run0": binaries[0] if len(binaries) > 0 else None,
            "run1": binaries[1] if len(binaries) > 1 else None,
            "run2": binaries[2] if len(binaries) > 2 else None,
            "pattern": pattern,
            "flips": flips,
        })
    return rows


def tab_run_consistency():
    st.header("Run Consistency (temp=0)")
    st.caption(
        "Same question, same wording, same model, temperature=0 — do the 3 runs give the same pass/fail? "
        "A flip means the model changed its answer across identical runs. "
        "Only genuine ✅↔❌ changes are counted — consistent wrong (❌❌❌) is not a flip."
    )

    rows = load_run_consistency()
    if not rows:
        st.info("No data yet.")
        return

    import pandas as pd
    df = pd.DataFrame(rows)
    flips_df = df[df["flips"]].copy()

    col1, col2, col3 = st.columns(3)
    col1.metric("Total (question, model) pairs", len(df))
    col2.metric("Flipping pairs", len(flips_df))
    col3.metric("Flip rate", f"{len(flips_df)/len(df):.1%}")

    # Pattern breakdown
    st.subheader("Flip patterns")
    pattern_counts = flips_df["pattern"].value_counts().reset_index()
    pattern_counts.columns = ["Pattern", "Count"]
    pattern_counts["Meaning"] = pattern_counts["Pattern"].map({
        "✅❌❌": "Got it right once then wrong",
        "❌✅✅": "Wrong first then corrected",
        "✅✅❌": "Right twice then wrong",
        "❌❌✅": "Wrong twice then right",
        "✅❌✅": "Alternating",
        "❌✅❌": "Alternating",
    }).fillna("Other")
    col_p, col_m = st.columns([1, 2])
    with col_p:
        st.dataframe(pattern_counts, use_container_width=True, hide_index=True)
    with col_m:
        fig = px.bar(pattern_counts, x="Pattern", y="Count", color="Count",
                     color_continuous_scale="Reds", text="Count",
                     title="How often does each flip pattern appear?")
        fig.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    # Which models flip most
    st.subheader("Which models flip most")
    model_flips = (
        df.groupby("model_display")
        .agg(total=("flips", "count"), flips=("flips", "sum"))
        .reset_index()
    )
    model_flips["flip_rate"] = model_flips["flips"] / model_flips["total"]
    model_flips = model_flips.sort_values("flip_rate", ascending=False)
    fig2 = px.bar(model_flips, y="model_display", x="flip_rate", orientation="h",
                  color="flip_rate", color_continuous_scale="Reds", range_color=[0, 0.3],
                  text=(model_flips["flip_rate"]*100).round(1).astype(str)+"%",
                  labels={"model_display": "", "flip_rate": "Flip Rate"},
                  title="Flip rate per model")
    fig2.update_traces(textposition="outside")
    fig2.update_layout(coloraxis_showscale=False, yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(fig2, use_container_width=True)

    # Browse all flip cases
    st.subheader("Browse flip cases")
    raw = load_raw(str(RAW_DIR))
    questions_map = {q["id"]: q for q in load_questions("factual")}
    true_variants, mostly_same = load_group_types()

    def group_type_badge(wg):
        if wg in true_variants: return "✓ wording variant"
        if wg in mostly_same:   return "~ mostly same"
        return "✗ topic cluster"

    def group_filter_label(wg):
        return f"{wg}  [{group_type_badge(wg)}]"

    col_f1, col_f2 = st.columns(2)
    with col_f1:
        model_filter = st.selectbox("Filter by model", ["All"] + sorted(flips_df["model_display"].unique()), key="rc_model")
    with col_f2:
        qg_options = sorted(flips_df["wording_group"].unique())
        qg_filter = st.selectbox("Filter by question", ["All"] + qg_options,
                                 format_func=lambda x: x if x == "All" else group_filter_label(x),
                                 key="rc_qgroup")

    filtered = flips_df
    if model_filter != "All":
        filtered = filtered[filtered["model_display"] == model_filter]
    if qg_filter != "All":
        filtered = filtered[filtered["wording_group"] == qg_filter]

    st.caption(f"Showing {len(filtered)} flip cases")

    for _, row in filtered.iterrows():
        badge = group_type_badge(row["wording_group"])
        with st.expander(
            f"**{row['question_id']}** [{badge}] — {row['model_display']} — {row['pattern']}",
            expanded=False,
        ):
            q = questions_map.get(row["question_id"])
            if q:
                st.info(f"_{q['prompt_pt']}_  \n**Ground truth:** {q['ground_truth']}")

            run_cols = st.columns(3)
            for i, (rc, run_num) in enumerate(zip(run_cols, [0, 1, 2])):
                with rc:
                    b = [row["run0"], row["run1"], row["run2"]][i]
                    chip = "✅" if (b is not None and b >= 0.5) else "❌"
                    st.markdown(f"**Run {run_num}** {chip}")
                    r = raw.get((row["question_id"], row["model"], run_num))
                    if r:
                        st.write((r.get("response") or "*no response recorded*")[:300])
                    else:
                        st.write("*not found*")


def tab_wording_comparison():
    st.header("Wording & Run Comparison")
    st.caption("Pick a question group and model to see all 3 wordings (A/B/C) and all 3 runs side by side.")

    questions = load_questions("factual")
    raw = load_raw(str(RAW_DIR))
    scores_df = load_scores()

    # Build lookup: wording_group → list of question dicts
    from collections import defaultdict
    groups = defaultdict(list)
    for q in questions:
        groups[q["wording_group"]].append(q)

    true_variants, mostly_same = load_group_types()

    def group_label(g):
        if g in true_variants:
            return f"{g}  ✓ true variant"
        if g in mostly_same:
            return f"{g}  ~ mostly same"
        return f"{g}  ✗ topic cluster"

    col1, col2 = st.columns(2)
    with col1:
        selected_group = st.selectbox(
            "Question group", sorted(groups.keys()),
            format_func=group_label,
            key="wr_qgroup",
        )
    with col2:
        selected_model = st.selectbox(
            "Model", sorted(MODEL_DISPLAY.keys()),
            format_func=lambda m: MODEL_DISPLAY.get(m, m),
            key="wr_model",
        )

    group_qs = sorted(groups[selected_group], key=lambda q: q["wording_variant"])

    if selected_group in true_variants:
        st.success("True wording variants — all 3 phrasings test the same underlying fact.")
    elif selected_group in mostly_same:
        st.warning("Mostly the same fact — 2 of 3 variants are true phrasings; one drifts slightly.")
    else:
        st.error("Topic cluster — variants ask different factual questions about the same subject. Not valid for wording sensitivity analysis.")

    # Show ground truth
    st.info(f"**Ground truth:** {group_qs[0]['ground_truth']}")

    # ── Wording variants side by side ────────────────────────────────────────
    st.subheader("Wording variants (A / B / C) — run 0")
    cols = st.columns(len(group_qs))
    for col, q in zip(cols, group_qs):
        with col:
            r = raw.get((q["id"], selected_model, 0))
            # get score for this specific response
            score_row = None
            if not scores_df.empty and "question_id" in scores_df.columns:
                match = scores_df[
                    (scores_df["question_id"] == q["id"]) &
                    (scores_df["model"] == selected_model) &
                    (scores_df.get("run", scores_df.get("run", 0)) == 0)
                ]
                if not match.empty:
                    score_row = match.iloc[0]

            b = score_row["binary_avg"] if score_row is not None and "binary_avg" in score_row else None
            chip = "✅" if b == 1.0 else ("❌" if b == 0.0 else "⚪")
            st.markdown(f"**Variant {q['wording_variant']}** {chip}")
            st.caption(q["prompt_pt"])
            st.write(r["response"] if r else "*no response*")
            if score_row is not None:
                with st.expander("Judge reasoning"):
                    render_judge_scores(score_row.to_dict() if hasattr(score_row, 'to_dict') else score_row)

    # ── Runs side by side (for each variant) ────────────────────────────────
    st.subheader("Are the 3 runs identical? (temp=0)")
    for q in group_qs:
        responses = [raw.get((q["id"], selected_model, run)) for run in range(3)]
        texts = [r["response"] if r else None for r in responses]
        valid = [t for t in texts if t]
        all_same = len(set(t[:200] for t in valid)) == 1 if len(valid) > 1 else True
        same_label = "identical" if all_same else "**differ**"
        with st.expander(f"Variant {q['wording_variant']} — runs 0/1/2 are {same_label}"):
            st.caption(q["prompt_pt"])
            run_cols = st.columns(3)
            for i, (rc, text) in enumerate(zip(run_cols, texts)):
                with rc:
                    st.markdown(f"**Run {i}**")
                    if text:
                        # highlight if different from run 0
                        if i > 0 and texts[0] and text[:200] != texts[0][:200]:
                            st.warning(text)
                        else:
                            st.write(text)
                    else:
                        st.write("*not found*")


# ── CONSISTENCY ───────────────────────────────────────────────────────────────

@st.cache_data(ttl=30)
def load_t10_scores():
    rows = []
    if not SCORES_T10_DIR.exists():
        return pd.DataFrame()
    for f in SCORES_T10_DIR.glob("*.json"):
        try:
            d = json.load(open(f))
            if not isinstance(d, dict):
                continue
            rows.append(d)
        except Exception:
            pass
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    if "model" in df.columns:
        df["model_display"] = df["model"].map(MODEL_DISPLAY).fillna(df["model"])
    return df


def tab_consistency():
    st.header("Consistency at Temperature 1.0")
    st.caption(
        "Same questions re-run 3× at temp=1.0 (real-world default). "
        "Flip rate = fraction of questions where binary verdict changes across runs."
    )

    raw_t10   = load_raw(str(RAW_T10_DIR))
    t10_scores = load_t10_scores()
    scores_df  = load_scores()

    if not raw_t10:
        st.info("No temp=1.0 data found. Run `python run.py --temperature 1.0 --runs 3` first.")
        return

    from collections import defaultdict

    # Flip rate from scored data (binary_avg per run)
    if not t10_scores.empty and "category" in t10_scores.columns:
        fdf10 = t10_scores[t10_scores["category"] == "factual"].copy()
        by_qm = defaultdict(list)
        for _, row in fdf10.iterrows():
            b = row.get("binary_avg")
            if b is not None:
                by_qm[(row["question_id"], row["model"])].append(b)
        model_flip = defaultdict(list)
        for (qid, model), bs in by_qm.items():
            if len(bs) >= 2:
                model_flip[model].append(1 if len(set(bs)) > 1 else 0)
        scored_flip = True
    else:
        # Fallback: text similarity
        flip_by_model = defaultdict(list)
        for (qid, model, run), d in raw_t10.items():
            flip_by_model[(qid, model)].append(d["response"][:120])
        model_flip = defaultdict(list)
        for (qid, model), responses in flip_by_model.items():
            if len(responses) >= 2:
                unique = len(set(responses))
                model_flip[model].append(1 if unique > 1 else 0)
        scored_flip = False

    if not scored_flip:
        st.warning("Scores not yet available — showing text-similarity flip rate (approximate).")

    # temp=0 and temp=1.0 accuracy
    t0_acc, t10_acc = {}, {}
    if not scores_df.empty and "category" in scores_df.columns:
        fdf0 = scores_df[scores_df["category"] == "factual"]
        if not fdf0.empty:
            t0_acc = fdf0.groupby("model")["binary_avg"].mean().to_dict()
    if not t10_scores.empty and "category" in t10_scores.columns:
        fdf10b = t10_scores[t10_scores["category"] == "factual"]
        if not fdf10b.empty:
            t10_acc = fdf10b.groupby("model")["binary_avg"].mean().to_dict()

    rows = []
    for model in sorted(model_flip):
        flips = model_flip[model]
        rows.append({
            "model": MODEL_DISPLAY.get(model, model),
            "flip_rate": round(sum(flips) / len(flips), 2),
            "t0_accuracy": round(t0_acc.get(model, 0), 2),
            "t10_accuracy": round(t10_acc.get(model, 0), 2),
            "n_questions": len(flips),
        })

    cdf = pd.DataFrame(rows).sort_values("flip_rate", ascending=False)

    col1, col2 = st.columns(2)
    with col1:
        fig = px.bar(
            cdf, y="model", x="flip_rate", orientation="h",
            color="flip_rate", color_continuous_scale="Reds", range_color=[0, 1],
            text=(cdf["flip_rate"] * 100).round(1).astype(str) + "%",
            title="Flip Rate — how often do 3 runs disagree?",
            labels={"model": "", "flip_rate": "Flip Rate"},
        )
        fig.update_traces(textposition="outside")
        fig.update_layout(coloraxis_showscale=False, height=500, yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig2 = px.scatter(
            cdf, x="t0_accuracy", y="flip_rate",
            text="model",
            hover_data={"t10_accuracy": True, "n_questions": True},
            title="Accuracy (temp=0) vs Flip Rate (temp=1.0)",
            labels={"t0_accuracy": "Accuracy at temp=0", "flip_rate": "Flip Rate at temp=1.0"},
            color="flip_rate", color_continuous_scale="RdYlGn_r",
        )
        fig2.update_traces(textposition="top center")
        fig2.update_layout(coloraxis_showscale=False, height=500)
        st.plotly_chart(fig2, use_container_width=True)

    # accuracy comparison table
    if t10_acc:
        st.subheader("Accuracy: temp=0 vs temp=1.0")
        cdf["accuracy_drop"] = (cdf["t0_accuracy"] - cdf["t10_accuracy"]).round(2)
        st.dataframe(
            cdf[["model", "t0_accuracy", "t10_accuracy", "accuracy_drop", "flip_rate"]]
            .rename(columns={
                "model": "Model",
                "t0_accuracy": "Acc (temp=0)",
                "t10_accuracy": "Acc (temp=1.0)",
                "accuracy_drop": "Drop",
                "flip_rate": "Flip Rate",
            })
            .sort_values("accuracy_drop", ascending=False),
            use_container_width=True, hide_index=True,
        )

    st.subheader("Side-by-side: same question, 3 runs at temp=1.0")
    questions = load_questions("factual")
    models_available = sorted(set(model for (_, model, _) in raw_t10))

    col_q, col_m = st.columns(2)
    with col_q:
        q_ids = sorted(set(q["id"] for q in questions))
        selected_q = st.selectbox("Question", q_ids, key="con_question")
    with col_m:
        selected_m = st.selectbox("Model", models_available, format_func=lambda m: MODEL_DISPLAY.get(m, m), key="con_model")

    runs = [raw_t10.get((selected_q, selected_m, r)) for r in range(3)]
    cols = st.columns(3)
    for i, (col, r) in enumerate(zip(cols, runs)):
        with col:
            st.markdown(f"**Run {i}**")
            st.write(r["response"] if r else "*not found*")


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    st.title("🇧🇷 Brazil Politics Eval")
    st.caption("Auditing AI model bias in Brazilian political discourse — 2026 presidential election.")

    tabs = st.tabs([
        "Factual Overview",
        "Wording & Runs",
        "Run Consistency",
        "Judge Disagreements",
        "Follow-up Explorer",
        "Consistency (temp=1.0)",
    ])

    df = load_scores()

    with tabs[0]:
        tab_factual_overview(df)
    with tabs[1]:
        tab_wording_comparison()
    with tabs[2]:
        tab_run_consistency()
    with tabs[3]:
        tab_disagreements()
    with tabs[4]:
        tab_followups()
    with tabs[5]:
        tab_consistency()


if __name__ == "__main__":
    main()
