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
DEMO_BUDGET = 6000           # zero-shot budget demo
FINAL_BUDGET = 100000        # few-shot production run
CATS = ["billing", "delivery", "technical", "fraud"]
CLASSIFIED = ("classified_json", "classified_keyword")
TO_HUMAN = ("skipped_empty", "flagged_long", "flagged_invalid_output", "budget_exhausted")
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


def assert_no_pii(text):
    for name, pat, _ in PII:
        assert not pat.search(text), f"{name} would reach the model"


def call_with_retry(messages, ticket_id):
    """Returns (response, retries_used). Re-raises after MAX_RETRIES failed retries."""
    for n in range(MAX_RETRIES + 1):
        try:
            return mock_llm.chat(messages), n
        except TransientLLMError as e:
            if n == MAX_RETRIES:
                raise
            wait = BASE_DELAY * 2 ** n * _jitter.uniform(0.9, 1.15)
            print(f"  [retry] ticket={ticket_id} retry={n} wait={wait:.4f}s error={e}")
            time.sleep(wait)


def validate(reply):
    """Returns (status, category or None). Never guesses."""
    try:
        data = json.loads(reply)
    except (json.JSONDecodeError, TypeError):
        found = {w for w in re.findall(r"[a-z]+", (reply or "").lower()) if w in CATS}
        return ("classified_keyword", found.pop()) if len(found) == 1 else ("flagged_invalid_output", None)
    if isinstance(data, dict) and data.get("category") in CATS:
        return "classified_json", data["category"]
    return "flagged_invalid_output", None


def run(tickets, system_prompt, label, budget=None):
    mock_llm.reset()
    print(f"\n--- run: {label} (budget={budget}) ---")
    rows, spent, errors = [], 0, 0
    for t in tickets.itertuples(index=False):
        row = {"ticket_id": t.ticket_id, "customer_id": t.customer_id, "true_category": t.true_category,
               "status": None, "category": None, "retries": 0, "tokens_used": 0}
        if t.text.strip() == "":
            row["status"] = "skipped_empty"
        elif est_tokens(t.text) > MAX_INPUT_TOKENS:
            row["status"] = "flagged_long"
        elif budget is not None and spent >= budget:
            row["status"] = "budget_exhausted"
        else:
            messages = [{"role": "system", "content": system_prompt},
                        {"role": "user", "content": t.redacted_text}]
            assert_no_pii(messages[-1]["content"])
            try:
                resp, row["retries"] = call_with_retry(messages, t.ticket_id)
            except TransientLLMError:
                row["status"] = "failed_permanent"
            else:
                errors += row["retries"]
                row["tokens_used"] = resp.usage.total_tokens
                spent += row["tokens_used"]
                row["status"], row["category"] = validate(resp.choices[0].message.content)
        rows.append(row)
    res = pd.DataFrame(rows)
    res["cumulative_tokens"] = res["tokens_used"].cumsum()
    print(f"transient errors recovered: {errors}; model calls: {int((res['tokens_used'] > 0).sum())}; "
          f"tokens spent: {spent}")
    print(res["status"].value_counts().to_string())
    assert (res["status"] == "failed_permanent").sum() == 0
    return res


def summarise(res, label):
    ok = res[res["status"].isin(CLASSIFIED)]
    correct = int((ok["category"] == ok["true_category"]).sum())
    tok = int(res["tokens_used"].sum())
    return {"prompt": label, "classified": len(ok), "correct": correct,
            "accuracy_pct": round(correct / len(ok) * 100, 1), "total_tokens": tok,
            "tokens_per_classified": round(tok / len(ok), 1)}


def route(status, category, is_premium):
    if status in TO_HUMAN:
        return "human_review", "NORMAL"
    if category == "fraud":
        return "security_team", "HIGH"
    if is_premium:
        return "premium_desk", "HIGH"
    return "support_queue", "NORMAL"


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
    print(f"system prompt size: zero-shot {est_tokens(ZERO_SHOT)} est. tokens, few-shot {est_tokens(FEW_SHOT)}")

    zero = run(tickets, ZERO_SHOT, "zero-shot")
    few = run(tickets, FEW_SHOT, "few-shot")
    comp = pd.DataFrame([summarise(zero, "zero-shot"), summarise(few, "few-shot")])
    print("\n=== zero-shot vs few-shot ===")
    print(comp.to_string(index=False))
    for name, r in (("zero-shot", zero), ("few-shot", few)):
        ok = r[r["status"].isin(CLASSIFIED)]
        print(f"\n=== {name} confusion (rows=true, cols=predicted) ===")
        print(pd.crosstab(ok["true_category"], ok["category"]).to_string())

    demo = run(tickets, ZERO_SHOT, "zero-shot budget demo", budget=DEMO_BUDGET)
    demo.insert(1, "budget", DEMO_BUDGET)
    os.makedirs(OUT, exist_ok=True)
    demo.drop(columns="true_category").to_csv(os.path.join(OUT, "triage_budget_demo.csv"), index=False)
    ex = demo.index[demo["status"] == "budget_exhausted"]
    print(f"budget demo: {len(ex)} tickets budget_exhausted; first at file row {int(ex[0]) + 1} "
          f"({demo.loc[ex[0], 'ticket_id']}) after {int(demo.loc[ex[0], 'cumulative_tokens'])} tokens")
    assert len(ex) >= 1

    final = run(tickets, FEW_SHOT, "few-shot final", budget=FINAL_BUDGET)
    seg = pd.read_csv(os.path.join(OUT, "customer_segments.csv"))[["customer_id", "segment_name", "is_premium"]]
    final = final.merge(seg, on="customer_id", how="left")
    final["is_premium"] = final["is_premium"].fillna(False).astype(bool)
    final[["route", "priority"]] = [route(s, c, p) for s, c, p in
                                    zip(final["status"], final["category"], final["is_premium"])]
    cols = ["ticket_id", "customer_id", "status", "category", "segment_name", "priority", "route", "tokens_used"]
    final[cols].to_csv(os.path.join(OUT, "triage_results.csv"), index=False)
    assert len(final) == 164
    print("\n=== routing (few-shot final run) ===")
    print(final.groupby(["route", "priority"]).size().rename("tickets").to_string())
    print(f"premium customers' tickets: {int(final['is_premium'].sum())}; customers without a segment row: "
          f"{int(final['segment_name'].isna().sum())}")
    print(f"\nwrote {OUT}/triage_results.csv and {OUT}/triage_budget_demo.csv")


if __name__ == "__main__":
    main()
