"""Part C -- support-ticket triage: input guardrails, PII redaction, zero/few-shot prompts, retries with
backoff, output validation, token budget, and segment-aware routing.
Run from the repository root (after Part B):  python src/part_c_triage.py"""
import json
import os
import random
import re
import sys
import time

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import mock_llm  # noqa: E402
from mock_llm import TransientLLMError, est_tokens  # noqa: E402

OUT = "outputs"
MAX_INPUT_TOKENS = 400
MAX_RETRIES = 3
BASE_DELAY = 0.05
TOKEN_BUDGET = 3000          # demo budget for one run, in est_tokens units
CATS = ["billing", "delivery", "technical", "fraud"]
PII = [("EMAIL", re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"), "[EMAIL]"),
       ("PHONE", re.compile(r"\b[6-9]\d{9}\b"), "[PHONE]"),
       ("ORDER_ID", re.compile(r"ORD-\d{6}"), "[ORDER_ID]")]

ZERO_SHOT = """You CLASSIFY customer support tickets for a quick-commerce grocery app.
Allowed categories: billing, delivery, technical, fraud.
Reply with JSON only, no other text, exactly in the form {"category": "...", "reason": "..."}
where category is one of the four allowed values and reason is a few words."""

FEW_SHOT = ZERO_SHOT + """
EXAMPLES:
Ticket: Someone used my card for an order I never placed, this is an unauthorized transaction.
{"category": "fraud", "reason": "unauthorized card use"}
Ticket: The app shows an error and freezes on the checkout page.
{"category": "technical", "reason": "app error at checkout"}
Ticket: I was charged twice for the same order, please fix the bill.
{"category": "billing", "reason": "double charge"}
Ticket: The rider has not arrived and my order is 40 minutes late.
{"category": "delivery", "reason": "late delivery"}"""

_jitter = random.Random(42)


def redact(text, counts=None):
    for name, pat, ph in PII:
        text, n = pat.subn(ph, text)
        if counts is not None:
            counts[name] += n
    return text


def call_with_retry(messages, ticket_id):
    """Returns (response, retries_used). Re-raises after MAX_RETRIES failed retries."""
    for n in range(MAX_RETRIES + 1):
        try:
            return mock_llm.chat(messages), n
        except TransientLLMError as e:
            if n == MAX_RETRIES:
                raise
            wait = BASE_DELAY * 2 ** n * _jitter.uniform(0.9, 1.15)
            print(f"  [retry] ticket={ticket_id} retry={n} wait={wait:.3f}s error={e}")
            time.sleep(wait)


def validate(reply):
    """Returns (status, category or None). Never guesses."""
    try:
        data = json.loads(reply)
        if isinstance(data, dict) and data.get("category") in CATS:
            return "classified_json", data["category"]
        return "flagged_invalid_output", None
    except (json.JSONDecodeError, TypeError):
        found = {w for w in re.findall(r"[a-z]+", (reply or "").lower()) if w in CATS}
        if len(found) == 1:
            return "classified_keyword", found.pop()
        return "flagged_invalid_output", None


def route(category, status, is_premium):
    """Queue, priority and response SLA following escalation_policy.txt."""
    if status not in ("classified_json", "classified_keyword"):
        return "human_review", "high" if is_premium else "normal", 1 if is_premium else 24
    if category == "fraud":
        return "security_team", "high", 1 if is_premium else 2
    queue = {"billing": "billing_desk", "delivery": "delivery_ops", "technical": "tech_support"}[category]
    return queue, "high" if is_premium else "normal", 1 if is_premium else 24


def run(tickets, system_prompt, label, budget=None):
    mock_llm.reset()
    print(f"\n--- run: {label} (budget={budget}) ---")
    rows, spent, errors = [], 0, 0
    for t in tickets.itertuples(index=False):
        row = {"ticket_id": t.ticket_id, "customer_id": t.customer_id, "true_category": t.true_category,
               "status": None, "category": None, "retries": 0, "tokens": 0}
        if t.text.strip() == "":
            row["status"] = "skipped_empty"
        elif est_tokens(t.text) > MAX_INPUT_TOKENS:
            row["status"] = "flagged_long"
        else:
            messages = [{"role": "system", "content": system_prompt},
                        {"role": "user", "content": t.redacted_text}]
            projected = sum(est_tokens(m["content"]) for m in messages)
            if budget is not None and spent + projected > budget:
                row["status"] = "skipped_budget"
            else:
                try:
                    resp, row["retries"] = call_with_retry(messages, t.ticket_id)
                except TransientLLMError:
                    row["status"] = "failed_permanent"
                else:
                    errors += row["retries"]
                    row["tokens"] = resp.usage.total_tokens
                    spent += row["tokens"]
                    row["status"], row["category"] = validate(resp.choices[0].message.content)
        rows.append(row)
    res = pd.DataFrame(rows)
    print(f"transient errors recovered: {errors}; tokens spent: {spent}")
    print(res["status"].value_counts().to_string())
    return res


def summarise(res, label):
    ok = res[res["status"].isin(["classified_json", "classified_keyword"])]
    acc = (ok["category"] == ok["true_category"]).mean()
    tok = int(res["tokens"].sum())
    return {"prompt": label, "classified": len(ok), "correct": int((ok["category"] == ok["true_category"]).sum()),
            "accuracy_pct": round(acc * 100, 1), "total_tokens": tok,
            "tokens_per_classified": round(tok / len(ok), 1)}


def main():
    tickets = pd.read_csv(os.path.join("data", "raw", "tickets.csv"), keep_default_na=False)
    empty = tickets["text"].str.strip() == ""
    long_ = tickets["text"].map(est_tokens) > MAX_INPUT_TOKENS
    print(f"tickets {len(tickets)}; empty {int(empty.sum())}; long {int(long_.sum())} "
          f"(est_tokens {sorted(set(tickets.loc[long_, 'text'].map(est_tokens)))})")

    counts = {name: 0 for name, _, _ in PII}
    tickets["redacted_text"] = tickets["text"].map(lambda s: redact(s, counts))
    print(f"redactions: {counts}; total {sum(counts.values())}")
    left = sum(len(p.findall(s)) for s in tickets["redacted_text"] for _, p, _ in PII)
    print(f"PII matches left after redaction: {left}")
    assert left == 0

    assert "CLASSIFY" in ZERO_SHOT and "EXAMPLES:" not in ZERO_SHOT
    assert "CLASSIFY" in FEW_SHOT and "\nEXAMPLES:\n" in FEW_SHOT and '"fraud"' in FEW_SHOT

    zero = run(tickets, ZERO_SHOT, "zero-shot")
    few = run(tickets, FEW_SHOT, "few-shot")
    for r in (zero, few):
        assert (r["status"] == "failed_permanent").sum() == 0
    comp = pd.DataFrame([summarise(zero, "zero-shot"), summarise(few, "few-shot")])
    print("\n=== zero-shot vs few-shot ===")
    print(comp.to_string(index=False))
    ok = zero[zero["category"].notna()]
    print("\n=== zero-shot confusion (rows=true, cols=predicted) ===")
    print(pd.crosstab(ok["true_category"], ok["category"]).to_string())
    okf = few[few["category"].notna()]
    print("\n=== few-shot confusion (rows=true, cols=predicted) ===")
    print(pd.crosstab(okf["true_category"], okf["category"]).to_string())

    seg = pd.read_csv(os.path.join(OUT, "customer_segments.csv"))[["customer_id", "segment_name", "is_premium"]]
    final = few.merge(seg, on="customer_id", how="left")
    final["segment_name"] = final["segment_name"].fillna("No completed orders")
    final["is_premium"] = final["is_premium"].fillna(False).astype(bool)
    final[["queue", "priority", "sla_hours"]] = [route(c, s, p) for c, s, p in
                                                 zip(final["category"], final["status"], final["is_premium"])]
    final = final.merge(tickets[["ticket_id", "redacted_text"]], on="ticket_id")
    final.insert(1, "prompt", "few-shot")
    os.makedirs(OUT, exist_ok=True)
    final.to_csv(os.path.join(OUT, "triage_results.csv"), index=False)
    print("\n=== routing (few-shot run) ===")
    print(final.groupby(["queue", "priority", "sla_hours"]).size().rename("tickets").to_string())
    print(f"premium tickets: {int(final['is_premium'].sum())}")

    demo = run(tickets, FEW_SHOT, "few-shot with budget", budget=TOKEN_BUDGET)
    demo.insert(1, "budget", TOKEN_BUDGET)
    demo["cumulative_tokens"] = demo["tokens"].cumsum()
    demo.to_csv(os.path.join(OUT, "triage_budget_demo.csv"), index=False)
    first_skip = demo.index[demo["status"] == "skipped_budget"]
    print(f"budget demo: spent {int(demo['tokens'].sum())} of {TOKEN_BUDGET}; first budget skip at row "
          f"{int(first_skip[0]) + 1 if len(first_skip) else None} of {len(demo)}")
    print(f"\nwrote {OUT}/triage_results.csv and {OUT}/triage_budget_demo.csv")


if __name__ == "__main__":
    main()
