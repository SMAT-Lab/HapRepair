#!/usr/bin/env python3
"""
LLM-assisted semantic labeling for HapRepair repairs using two independent judges.

This script reads a prepared semantic-eval sample produced by:
  - revision/code/repair_semantic_prepare.py         (finding-level)
  - revision/code/repair_semantic_prepare_files.py   (file-level)

and asks two LLM judges to label each repair as:
  - Correct
  - Suspicious
  - Incorrect

It writes:
  - llm_judgments.jsonl   (per-sample raw + parsed judgments)
  - labels_llm.csv        (labels in the format expected by repair_semantic_summarize.py)

Judge routing (project conventions):
  - gpt-5.1      : Packy (handled inside /home/LLMCodeRepair/llm.py)
  - deepseek-chat: Zhizengzeng (set API_KEY/API_BASE from ZHIZENGZENG_* in .env)

IMPORTANT: This is NOT human evaluation. If used in the paper, it must be
disclosed as LLM-judged / LLM-assisted evaluation.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
import concurrent.futures
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


REPO_ROOT = Path("/home/LLMCodeRepair").resolve()
DEFAULT_ENV = REPO_ROOT / ".env"

# Ensure we can import /home/LLMCodeRepair/llm.py when running from other CWDs.
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _load_dotenv(path: Path) -> Dict[str, str]:
    if not path.is_file():
        return {}
    env: Dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        env[k.strip()] = v.strip().strip('"').strip("'")
    return env


def _ensure_zhizengzeng_env() -> None:
    """
    llm.py uses API_KEY/API_BASE for non-packy models.
    For deepseek-chat we route via zhizengzeng by mapping ZHIZENGZENG_* to API_*.
    """
    if os.getenv("API_KEY") and os.getenv("API_BASE"):
        return

    file_env = _load_dotenv(DEFAULT_ENV)
    zh_key = os.getenv("ZHIZENGZENG_API_KEY") or file_env.get("ZHIZENGZENG_API_KEY")
    zh_base = os.getenv("ZHIZENGZENG_API_BASE") or file_env.get("ZHIZENGZENG_API_BASE")
    if zh_key and not os.getenv("API_KEY"):
        os.environ["API_KEY"] = zh_key
    if zh_base and not os.getenv("API_BASE"):
        os.environ["API_BASE"] = zh_base


JSON_BLOCK_RE = re.compile(r"```json\s*(?P<body>.*?)\s*```", re.DOTALL | re.IGNORECASE)


def _parse_jsonish(text: str) -> Optional[Dict[str, Any]]:
    text = (text or "").strip()
    if not text:
        return None
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except Exception:
        pass
    m = JSON_BLOCK_RE.search(text)
    if m:
        body = m.group("body").strip()
        try:
            obj = json.loads(body)
            return obj if isinstance(obj, dict) else None
        except Exception:
            return None
    return None


def _normalize_label(label: str) -> Optional[str]:
    l = (label or "").strip().lower()
    if l in ("correct", "likelycorrect", "likely_correct", "likely-correct", "✅ correct", "ok", "yes"):
        return "Correct"
    if l in ("suspicious", "uncertain", "unknown", "maybe", "⚠️ suspicious"):
        return "Suspicious"
    if l in ("incorrect", "wrong", "no", "❌ incorrect"):
        return "Incorrect"
    return None


def _clamp01(x: Any) -> Optional[float]:
    try:
        v = float(x)
    except Exception:
        return None
    if v < 0.0:
        return 0.0
    if v > 1.0:
        return 1.0
    return v


def _judge_prompt(sample: Dict[str, Any], *, prompt_mode: str) -> str:
    before_file = sample.get("before_file_text")
    after_file = sample.get("after_file_text")
    fixed_findings = sample.get("fixed_findings")
    file_diff = sample.get("file_diff")
    if isinstance(before_file, str) and isinstance(after_file, str) and isinstance(fixed_findings, list):
        semantic_only = prompt_mode == "file_semantic_only"
        findings_lines: List[str] = []
        for i, d in enumerate(fixed_findings[:200], 1):
            if not isinstance(d, dict):
                continue
            findings_lines.append(
                f"{i}. {d.get('category')}/{d.get('rule_id')} "
                f"({d.get('severity')}) at {d.get('line')}:{d.get('column')}: {d.get('message')}"
            )
        if len(fixed_findings) > 200:
            findings_lines.append(f"... ({len(fixed_findings) - 200} more findings omitted)")

        checks_block = [
            "Judge the repair using two checks (manual-review style):",
            "1) Rule issue remaining: does the AFTER file still *obviously* show the same issue pattern targeted by the fixed findings?",
            "2) Functional regression/omission: besides addressing the findings, did the patch omit original functionality or introduce a clear behavior regression?",
            "",
        ]
        out_schema = '{ "rule_issue_remaining": "yes|no|unclear", "functional_regression": "yes|no|unclear", "confidence": 0.0, "rationale": "one sentence" }'
        if semantic_only:
            checks_block = [
                "Judge the repair using one check (semantic-only):",
                "1) Functional regression/omission: did the patch omit original functionality or introduce a clear behavior regression?",
                "",
            ]
            out_schema = '{ "functional_regression": "yes|no|unclear", "confidence": 0.0, "rationale": "one sentence" }'

        return "\n".join(
            [
                "You are evaluating whether a repaired file is semantically correct.",
                "",
                "You are given:",
                "1) BEFORE (original file)",
                "2) AFTER (repaired file)",
                "3) The list of static findings that were fixed in this file",
                "",
                *checks_block,
                "Guidance (align with practical manual review):",
                "- Focus primarily on the DIFF and the specific changed regions; you do not need to read the entire file end-to-end.",
                "- Answer 'yes' only if you can point to specific code evidence in the DIFF / AFTER file.",
                "- Default to 'no' when you do not see evidence of a remaining rule issue or unrelated logic change.",
                "- Use 'unclear' only when the provided text is clearly insufficient (e.g., truncated) or the change is a large refactor where you truly cannot judge.",
                "",
                "Assumptions (important):",
                "- The static checker is high-precision and the AFTER version has passed re-scanning; treat the listed findings as genuinely fixed (not false positives).",
                "- Ignore syntax/compilation concerns. Assume the AFTER version type-checks and builds successfully; do NOT mark issues like 'missing import file' or 'undefined symbol' unless the semantic behavior is clearly wrong even under that assumption.",
                "",
                "Important: extra edits are allowed. Do NOT mark regression just because the patch makes additional safe changes (formatting, renaming, hoisting constants, changing enum literal casing, adding helper calls inside callbacks).",
                "Only mark functional regression/omission if you can show that some behavior/state update present BEFORE is missing or meaningfully weakened in AFTER.",
                "",
                "Examples of changes that are considered IN-SCOPE (do NOT mark as 'logic_changed=yes' just for these):",
                "- Replacing @State temporary variables with local variables (including accumulating a string locally and assigning to a state field at the end).",
                "- Replacing repeated `new Date().getTime()` with `Date.now()`.",
                "- Removing commented-out code / redundant UI wrappers.",
                "- Hoisting constant property access out of loops.",
                "- Adding `.onAnimationStart(...)` for Swiper preload as requested by the rule.",
                "",
                "Focus: (1) does the AFTER still obviously exhibit the same issue pattern, and (2) is there clear functionality omission/regression beyond what the fix needs.",
                "",
                "Output JSON only (no markdown):",
                out_schema,
                "",
                "File metadata:",
                f"- project: {sample.get('project')}",
                f"- round_fixed: {sample.get('round_fixed')} ({sample.get('stage_bin')})",
                f"- file: {sample.get('rel_path')}",
                f"- fixed_findings_count: {len(fixed_findings)}",
                "",
                "Fixed findings (disappeared after repair):",
                *findings_lines,
                "",
                "DIFF (before -> after):",
                file_diff if isinstance(file_diff, str) else "",
                "",
                "BEFORE (full file):",
                before_file,
                "",
                "AFTER (full file):",
                after_file,
            ]
        )

    return "\n".join(
        [
            "You are evaluating whether a code repair is semantically correct.",
            "You will be given local BEFORE/AFTER code contexts and a local diff.",
            "",
            "Judge the repair using three checks (as a careful human reviewer would):",
            "1) Semantic equivalence: is functionality preserved, or clearly broken?",
            "2) Rule intent: does the fix satisfy the rule's intent (not a hack)?",
            "3) New issues: does it introduce an obvious new bug (compile error, broken chain call, wrong API usage)?",
            "",
            "Be conservative: if you cannot determine a check from the given evidence, answer 'unclear'.",
            "",
            "Output JSON only (no markdown):",
            '{ "semantic_equivalent": "yes|no|unclear", "rule_intent_satisfied": "yes|no|unclear", "introduces_new_issue": "yes|no|unclear", "confidence": 0.0, "rationale": "one sentence" }',
            "",
            "Static finding metadata:",
            f"- project: {sample.get('project')}",
            f"- round_fixed: {sample.get('round_fixed')} ({sample.get('stage_bin')})",
            f"- rule: {sample.get('category')}/{sample.get('rule_id')} ({sample.get('severity')})",
            f"- message: {sample.get('message')}",
            f"- file: {sample.get('rel_path')}:{sample.get('line')}:{sample.get('column')}",
            "",
            "BEFORE (local context):",
            sample.get("before_context") or "",
            "",
            "AFTER (local context):",
            sample.get("after_context") or "",
            "",
            "LOCAL DIFF:",
            sample.get("local_diff") or "",
        ]
    )


@dataclass
class JudgeResult:
    model: str
    raw: str
    label: Optional[str]
    rule_issue_remaining: Optional[str]  # yes|no|unclear
    functional_regression: Optional[str]  # yes|no|unclear
    confidence: Optional[float]
    rationale: str


def _judge_risk(jr: JudgeResult, *, final_criteria: str) -> str:
    """
    Map judge output to a coarse risk bucket:
      - incorrect: rule issue remains OR functional regression (default)
      - correct: no rule issue AND no functional regression (default)
      - unclear: anything else (missing evidence or undecidable)
    """
    if final_criteria == "regression_only":
        if jr.functional_regression == "yes":
            return "incorrect"
        if jr.functional_regression == "no":
            return "correct"
        if jr.functional_regression is None:
            # Backward-compatible fallback when tri-fields are missing.
            if jr.label == "Incorrect":
                return "incorrect"
            if jr.label == "Correct":
                return "correct"
        return "unclear"

    if jr.rule_issue_remaining == "yes" or jr.functional_regression == "yes":
        return "incorrect"
    if jr.rule_issue_remaining == "no" and jr.functional_regression == "no":
        return "correct"
    # Backward-compatible fallback when tri-fields are missing.
    if jr.rule_issue_remaining is None and jr.functional_regression is None:
        if jr.label == "Incorrect":
            return "incorrect"
        if jr.label == "Correct":
            return "correct"
    return "unclear"


def _call_llm(model: str, prompt: str, *, dry_run: bool = False) -> str:
    if dry_run:
        # Deterministic fake response for pipeline testing.
        return json.dumps(
            {
                "rule_issue_remaining": "unclear",
                "functional_regression": "unclear",
                "confidence": 0.5,
                "rationale": f"dry-run stub for {model}",
            }
        )

    # Import lazily so dry-run doesn't require network/client config.
    from llm import get_answer  # type: ignore

    if model == "deepseek-chat":
        _ensure_zhizengzeng_env()
    return get_answer(prompt, model_name=model, system_prompt=None)


def _run_judge(model: str, prompt: str, *, dry_run: bool = False) -> JudgeResult:
    raw = _call_llm(model, prompt, dry_run=dry_run)
    parsed = _parse_jsonish(raw)
    if not parsed:
        return JudgeResult(
            model=model,
            raw=raw,
            label=None,
            rule_issue_remaining=None,
            functional_regression=None,
            confidence=None,
            rationale="unparseable",
        )

    def _norm_tri_file(v: Any) -> Optional[str]:
        s = str(v or "").strip().lower()
        if s in ("yes", "y", "true"):
            return "yes"
        if s in ("no", "n", "false"):
            return "no"
        if s in ("unclear", "unknown", "unsure", "maybe", "cannot_tell", "needs_runtime", "abstain"):
            return "unclear"
        return None

    def _norm_tri_generic(v: Any) -> Optional[str]:
        s = str(v or "").strip().lower()
        if s in ("yes", "y", "true", "preserved", "equivalent", "ok", "satisfied"):
            return "yes"
        if s in ("no", "n", "false", "broken", "changed", "not_equivalent", "violate", "not_satisfied"):
            return "no"
        if s in ("unclear", "unknown", "unsure", "maybe", "cannot_tell", "needs_runtime", "abstain"):
            return "unclear"
        return None

    rule_issue_remaining = _norm_tri_file(parsed.get("rule_issue_remaining"))
    functional_regression = _norm_tri_file(parsed.get("functional_regression"))
    # Backward compatibility for older file-level schema.
    if functional_regression is None:
        functional_regression = _norm_tri_file(parsed.get("logic_changed"))

    # Backward compatibility: accept a direct label if provided.
    label = _normalize_label(str(parsed.get("label", "")))
    if not label:
        # Support older (finding-level) schema.
        semantic_equivalent = _norm_tri_generic(parsed.get("semantic_equivalent"))
        rule_intent_satisfied = _norm_tri_generic(parsed.get("rule_intent_satisfied"))
        introduces_new_issue = _norm_tri_generic(parsed.get("introduces_new_issue"))

        if semantic_equivalent or rule_intent_satisfied or introduces_new_issue:
            if semantic_equivalent == "no" or rule_intent_satisfied == "no" or introduces_new_issue == "yes":
                label = "Incorrect"
            elif semantic_equivalent == "yes" and rule_intent_satisfied == "yes" and introduces_new_issue == "no":
                label = "Correct"
            else:
                label = "Suspicious"
        else:
            if rule_issue_remaining == "yes" or functional_regression == "yes":
                label = "Incorrect"
            elif rule_issue_remaining == "no" and functional_regression == "no":
                label = "Correct"
            else:
                label = "Suspicious"

    conf = _clamp01(parsed.get("confidence"))
    rationale = str(parsed.get("rationale", "")).strip()
    if not rationale:
        rationale = "no rationale"
    return JudgeResult(
        model=model,
        raw=raw,
        label=label,
        rule_issue_remaining=rule_issue_remaining,
        functional_regression=functional_regression,
        confidence=conf,
        rationale=rationale,
    )


def _combine(
    a: JudgeResult,
    b: JudgeResult,
    disagreement_policy: str,
    unclear_policy: str,
    final_criteria: str,
) -> Tuple[str, str]:
    """
    Return (final_label, notes).
    Notes is a compact summary stored in labels_llm.csv.
    """
    ra = _judge_risk(a, final_criteria=final_criteria)
    rb = _judge_risk(b, final_criteria=final_criteria)

    def _conf(jr: JudgeResult) -> float:
        return float(jr.confidence or 0.0)

    def _follow_by_confidence(j1: JudgeResult, j2: JudgeResult) -> Optional[str]:
        """
        Return a label to follow when judges conflict.
        Uses a confidence gap; otherwise returns None.
        """
        c1 = _conf(j1)
        c2 = _conf(j2)
        if abs(c1 - c2) >= 0.15:
            return j1.label if c1 > c2 else j2.label
        return None

    # If both are clearly correct/incorrect, take it.
    if ra == "correct" and rb == "correct":
        return "Correct", f"risk->Correct: {a.model}={ra}, {b.model}={rb}"
    if ra == "incorrect" and rb == "incorrect":
        return "Incorrect", f"risk->Incorrect: {a.model}={ra}({a.label}), {b.model}={rb}({b.label})"

    # If either judge failed to parse, keep conservative.
    if not a.label or not b.label:
        return "Suspicious", f"parse_issue: {a.model}={a.label}, {b.model}={b.label}"

    # Conflict: incorrect vs correct (common with different reviewer strictness).
    if (ra, rb) in (("incorrect", "correct"), ("correct", "incorrect")):
        if disagreement_policy == "confidence":
            chosen = _follow_by_confidence(a, b)
            if chosen:
                return chosen, f"conflict->follow_confidence: {a.model}={a.label}({a.confidence}), {b.model}={b.label}({b.confidence})"
            # Default optimistic under unclear_policy=likely_correct (align with "toolchain already passed").
            if unclear_policy == "likely_correct":
                return "Correct", f"conflict->LikelyCorrect: {a.model}={a.label}({a.confidence}), {b.model}={b.label}({b.confidence})"
        return "Suspicious", f"conflict->Suspicious: {a.model}={a.label}({a.confidence}), {b.model}={b.label}({b.confidence})"

    # If both are unclear, allow an optimistic mapping if requested.
    if ra == "unclear" and rb == "unclear" and unclear_policy == "likely_correct":
        return "Correct", f"unclear->LikelyCorrect: {a.model}={ra}({a.confidence}), {b.model}={rb}({b.confidence})"

    # If one is correct and the other unclear, allow optimistic mapping.
    if ((ra == "correct" and rb == "unclear") or (ra == "unclear" and rb == "correct")) and unclear_policy == "likely_correct":
        return "Correct", f"mixed->LikelyCorrect: {a.model}={ra}({a.confidence}), {b.model}={rb}({b.confidence})"

    # Remaining incorrect cases: incorrect vs unclear.
    if (ra, rb) in (("incorrect", "unclear"), ("unclear", "incorrect")):
        if disagreement_policy == "confidence":
            chosen = _follow_by_confidence(a, b)
            if chosen:
                return chosen, f"mixed->follow_confidence: {a.model}={a.label}({a.confidence}), {b.model}={b.label}({b.confidence})"
            if unclear_policy == "likely_correct":
                return "Correct", f"mixed->LikelyCorrect: {a.model}={ra}({a.confidence}), {b.model}={rb}({b.confidence})"
        return "Suspicious", f"mixed->Suspicious: {a.model}={ra}({a.confidence}), {b.model}={rb}({b.confidence})"

    if disagreement_policy == "incorrect_if_any":
        if "Incorrect" in (a.label, b.label):
            return "Incorrect", f"disagree->Incorrect: {a.model}={a.label}, {b.model}={b.label}"
        return "Suspicious", f"disagree->Suspicious: {a.model}={a.label}, {b.model}={b.label}"

    if disagreement_policy == "confidence":
        # If one judge is decisive with high confidence and the other is
        # Suspicious/unclear with low confidence, follow the decisive judge.
        decisive = None
        abstain = None
        for jr in (a, b):
            if jr.label in ("Correct", "Incorrect") and (jr.confidence or 0.0) >= 0.75:
                decisive = jr
            if jr.label == "Suspicious" and (jr.confidence or 0.0) <= 0.35:
                abstain = jr
        if decisive and abstain:
            return decisive.label, f"disagree->follow_confident_{decisive.model}: {a.model}={a.label}({a.confidence}), {b.model}={b.label}({b.confidence})"
        return "Suspicious", f"disagree->Suspicious: {a.model}={a.label}({a.confidence}), {b.model}={b.label}({b.confidence})"

    # Default: disagreements become Suspicious (forces conservative reporting).
    return "Suspicious", f"disagree->Suspicious: {a.model}={a.label}, {b.model}={b.label}"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Label semantic correctness with two LLM judges.")
    ap.add_argument("--run-dir", type=Path, required=True, help="semantic_eval run directory containing sample.jsonl.")
    ap.add_argument("--judge-a", type=str, default="gpt-5.1", help="First judge model (default: gpt-5.1).")
    ap.add_argument("--judge-b", type=str, default="deepseek-chat", help="Second judge model (default: deepseek-chat).")
    ap.add_argument(
        "--prompt-mode",
        choices=["file_default", "file_semantic_only"],
        default="file_default",
        help="Prompt rubric for file-level judging (default: file_default).",
    )
    ap.add_argument(
        "--final-criteria",
        choices=["rule_or_regression", "regression_only"],
        default="rule_or_regression",
        help="How to map judge fields to final labels (default: rule_or_regression).",
    )
    ap.add_argument(
        "--max-workers",
        type=int,
        default=64,
        help="Maximum parallel workers across samples (default: 64).",
    )
    ap.add_argument(
        "--disagreement-policy",
        choices=["suspicious", "incorrect_if_any", "confidence"],
        default="suspicious",
        help="How to resolve judge disagreements (default: suspicious).",
    )
    ap.add_argument(
        "--unclear-policy",
        choices=["suspicious", "likely_correct"],
        default="suspicious",
        help="How to treat unclear cases when there is no explicit evidence of incorrectness (default: suspicious).",
    )
    ap.add_argument("--max-items", type=int, default=0, help="If >0, only label first N items.")
    ap.add_argument("--sleep-sec", type=float, default=0.0, help="Sleep between API calls (default: 0).")
    ap.add_argument("--dry-run", action="store_true", help="Do not call APIs; emit stub judgments.")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    sample_path = run_dir / "sample.jsonl"
    if not sample_path.is_file():
        raise SystemExit(f"sample.jsonl not found: {sample_path}")

    raw_samples = _read_jsonl(sample_path)
    if args.max_items and args.max_items > 0:
        raw_samples = raw_samples[: args.max_items]

    # Only keep samples with stable ids; preserve order for reproducibility.
    samples: List[Dict[str, Any]] = []
    for s in raw_samples:
        sid = str(s.get("candidate_id") or "")
        if sid:
            samples.append(s)

    out_jsonl = run_dir / "llm_judgments.jsonl"
    out_csv = run_dir / "labels_llm.csv"

    def process_one(s: Dict[str, Any]) -> Dict[str, Any]:
        sid = str(s.get("candidate_id") or "")
        prompt = _judge_prompt(s, prompt_mode=args.prompt_mode)
        r1 = _run_judge(args.judge_a, prompt, dry_run=args.dry_run)
        if args.sleep_sec:
            time.sleep(args.sleep_sec)
        r2 = _run_judge(args.judge_b, prompt, dry_run=args.dry_run)
        final_label, notes = _combine(
            r1,
            r2,
            disagreement_policy=args.disagreement_policy,
            unclear_policy=args.unclear_policy,
            final_criteria=args.final_criteria,
        )
        return {
            "sample_id": sid,
            "project": s.get("project"),
            "round_fixed": s.get("round_fixed"),
            "category": s.get("category"),
            "rule_id": s.get("rule_id"),
            "rel_path": s.get("rel_path"),
            "line": s.get("line"),
            "column": s.get("column"),
            "judge_a": {
                "model": r1.model,
                "label": r1.label,
                "rule_issue_remaining": r1.rule_issue_remaining,
                "functional_regression": r1.functional_regression,
                "confidence": r1.confidence,
                "rationale": r1.rationale,
                "raw": r1.raw,
            },
            "judge_b": {
                "model": r2.model,
                "label": r2.label,
                "rule_issue_remaining": r2.rule_issue_remaining,
                "functional_regression": r2.functional_regression,
                "confidence": r2.confidence,
                "rationale": r2.rationale,
                "raw": r2.raw,
            },
            "final_label": final_label,
            "final_notes": notes,
            "final_criteria": args.final_criteria,
            "prompt_mode": args.prompt_mode,
        }

    out_jsonl.parent.mkdir(parents=True, exist_ok=True)

    # Parallel labeling across samples.
    results_by_id: Dict[str, Dict[str, Any]] = {}
    failures: List[Tuple[str, str]] = []
    max_workers = max(1, int(args.max_workers))
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs: Dict[concurrent.futures.Future[Dict[str, Any]], str] = {}
        for s in samples:
            sid = str(s.get("candidate_id") or "")
            futs[ex.submit(process_one, s)] = sid

        for fut in concurrent.futures.as_completed(futs):
            sid = futs[fut]
            try:
                rec = fut.result()
            except Exception as exc:
                failures.append((sid, f"{type(exc).__name__}: {exc}"))
                rec = {
                    "sample_id": sid,
                    "final_label": "Suspicious",
                    "final_notes": f"exception: {type(exc).__name__}",
                    "judge_a": {
                        "model": args.judge_a,
                        "label": None,
                        "rule_issue_remaining": None,
                        "functional_regression": None,
                        "confidence": None,
                        "rationale": "exception",
                        "raw": "",
                    },
                    "judge_b": {
                        "model": args.judge_b,
                        "label": None,
                        "rule_issue_remaining": None,
                        "functional_regression": None,
                        "confidence": None,
                        "rationale": "exception",
                        "raw": "",
                    },
                }
            results_by_id[sid] = rec

    # Write outputs in the original sample order for reproducibility.
    agreement = 0
    total = 0
    parse_fail = 0
    label_counts: Dict[str, int] = {"Correct": 0, "Suspicious": 0, "Incorrect": 0}

    with out_jsonl.open("w", encoding="utf-8") as jf, out_csv.open("w", encoding="utf-8", newline="") as cf:
        writer = csv.DictWriter(
            cf,
            fieldnames=[
                "sample_id",
                "label",
                "annotator",
                "notes",
                "project",
                "round_fixed",
                "category",
                "rule_id",
                "rel_path",
                "line",
            ],
        )
        writer.writeheader()

        for s in samples:
            sid = str(s.get("candidate_id") or "")
            rec = results_by_id.get(sid)
            if not rec:
                continue

            r1 = rec.get("judge_a") or {}
            r2 = rec.get("judge_b") or {}
            final_label = str(rec.get("final_label") or "Suspicious")
            notes = str(rec.get("final_notes") or "")

            if not r1.get("label") or not r2.get("label"):
                parse_fail += 1
            if r1.get("label") and r2.get("label") and r1.get("label") == r2.get("label"):
                agreement += 1
            total += 1
            label_counts[final_label] = label_counts.get(final_label, 0) + 1

            jf.write(json.dumps(rec, ensure_ascii=False) + "\n")
            writer.writerow(
                {
                    "sample_id": sid,
                    "label": final_label,
                    "annotator": f"llm:{args.judge_a}+{args.judge_b}",
                    "notes": notes,
                    "project": s.get("project"),
                    "round_fixed": s.get("round_fixed"),
                    "category": s.get("category"),
                    "rule_id": s.get("rule_id"),
                    "rel_path": s.get("rel_path"),
                    "line": s.get("line"),
                }
            )

    agree_rate = (agreement / total) if total else 0.0
    print(f"[ok] wrote: {out_jsonl}")
    print(f"[ok] wrote: {out_csv}")
    print(f"[stats] items={total}, agreement={agreement} ({agree_rate:.1%}), parse_fail={parse_fail}")
    print(f"[stats] label_counts={label_counts}")
    if failures:
        print(f"[warn] failures={len(failures)} (all mapped to Suspicious); first={failures[0]}")


if __name__ == "__main__":
    main()
