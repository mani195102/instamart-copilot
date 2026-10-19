"""Part D -- refund agent (ReAct loop over mock_llm AGENT) with tools backed by data/instamart.db and the policy
retriever, hard guardrails, a step limit, a token budget, an idempotent refund ledger, and checkpoint/resume.
Run from the repository root (after Part A):  python src/part_d_agent.py"""
import json
import os
import shutil
import sqlite3
import sys
import time

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mock_llm  # noqa: E402
from mock_llm import est_tokens  # noqa: E402
from policy_rag import PolicyIndex  # noqa: E402

DB = os.path.join("data", "instamart.db")
OUT, STATE = "outputs", "state"
RUNS, LEDGER = os.path.join(OUT, "agent_runs.jsonl"), os.path.join(OUT, "refund_ledger.jsonl")
MAX_STEPS = 6                  # escalation_policy.txt: hand over to a human after 6 steps
TOKEN_BUDGET = 4000            # per run, sum of response.usage.total_tokens
AUTO_LIMIT_INR = 1000          # refund_policy.txt: above this a human lead must approve
LATE_MINUTES = 45
SYSTEM = ("You are the Instamart refund AGENT. Use the tools to look up the order, check the refund policy, "
          "compute the refund and then either issue it or escalate it. Never refund more than the policy allows.")
TOOLS = [{"name": "get_order", "parameters": {"order_id": "string"}},
         {"name": "search_policy", "parameters": {"query": "string"}},
         {"name": "calculate_refund", "parameters": {"order_amount": "number", "delivery_minutes": "number"}},
         {"name": "issue_refund", "parameters": {"order_id": "string", "refund_inr": "number"}},
         {"name": "escalate_to_human", "parameters": {"order_id": "string", "refund_inr": "number"}}]

_index = None


class SimulatedCrash(Exception):
    """Injected process crash used by the crash-and-resume demonstration."""


# ------------------------------------------------------------------ ledger
def read_ledger():
    if not os.path.exists(LEDGER):
        return []
    with open(LEDGER, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def append_jsonl(path, obj):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj) + "\n")


# ------------------------------------------------------------------ tools
class Tools:
    """Tool implementations. The agent's customer_id is bound here, so a tool can never act for another customer."""

    def __init__(self, customer_id, run_id, db_path=DB):
        self.customer_id, self.run_id, self.db_path = customer_id, run_id, db_path
        self.verified = {}                 # order_id -> order dict fetched in this run

    def get_order(self, order_id):
        with sqlite3.connect(f"file:{self.db_path}?mode=ro", uri=True) as con:
            row = con.execute("SELECT order_id, customer_id, status, order_amount, delivery_minutes, order_ts "
                              "FROM orders WHERE order_id = ?", (order_id,)).fetchone()
        if row is None or row[1] != self.customer_id:      # unknown order, or not this customer's order
            return {"error": "order not found for this customer"}
        o = dict(zip(["order_id", "customer_id", "status", "order_amount", "delivery_minutes", "order_ts"], row))
        self.verified[order_id] = o
        return o

    def search_policy(self, query):
        global _index
        _index = _index or PolicyIndex()
        return {"results": [{"doc": h["doc"], "text": h["text"]} for h in _index.retrieve(query)]}

    def calculate_refund(self, order_amount, delivery_minutes):
        late = delivery_minutes is not None and delivery_minutes > LATE_MINUTES
        return {"refund_inr": round(float(order_amount), 2) if late else 0.0,
                "rule": f"full refund only if delivered in more than {LATE_MINUTES} minutes"}

    def _check(self, order_id, refund_inr):
        """Server-side guardrail: re-derive the allowed refund from the database, ignore what the model claims."""
        o = self.verified.get(order_id) or self.get_order(order_id)
        if "error" in o:
            return None, "order not verified"
        if o["status"] != "completed":
            return None, "only completed orders can be refunded"
        allowed = self.calculate_refund(o["order_amount"], o["delivery_minutes"])["refund_inr"]
        if allowed <= 0 or abs(float(refund_inr) - allowed) > 0.005:
            return None, f"requested INR {refund_inr} but policy allows INR {allowed}"
        return allowed, None

    def issue_refund(self, order_id, refund_inr):
        allowed, err = self._check(order_id, refund_inr)
        if err:
            return {"error": err}
        if allowed > AUTO_LIMIT_INR:
            return {"error": f"refund above INR {AUTO_LIMIT_INR} needs a human lead"}
        done = [e for e in read_ledger() if e["order_id"] == order_id and e["action"] == "refund_issued"]
        if done:                                              # idempotency: never pay the same order twice
            return {"refund_inr": done[0]["refund_inr"], "status": "already_refunded"}
        append_jsonl(LEDGER, {"ts": time.strftime("%Y-%m-%d %H:%M:%S"), "run_id": self.run_id,
                              "order_id": order_id, "customer_id": self.customer_id,
                              "action": "refund_issued", "refund_inr": allowed})
        return {"refund_inr": allowed, "status": "issued"}

    def escalate_to_human(self, order_id, refund_inr):
        allowed, err = self._check(order_id, refund_inr)
        if err:
            return {"error": err}
        done = [e for e in read_ledger() if e["order_id"] == order_id and e["action"] == "escalated"]
        if not done:
            append_jsonl(LEDGER, {"ts": time.strftime("%Y-%m-%d %H:%M:%S"), "run_id": self.run_id,
                                  "order_id": order_id, "customer_id": self.customer_id,
                                  "action": "escalated", "refund_inr": allowed})
        return {"refund_inr": allowed, "status": "escalated"}

    def execute(self, name, args):
        if name not in {t["name"] for t in TOOLS}:
            return {"error": f"unknown tool {name}"}
        try:
            return getattr(self, name)(**args)
        except Exception as e:                               # an infrastructure failure is reported, not raised
            return {"error": f"{type(e).__name__}: {e}", "tool_failure": True}


# ------------------------------------------------------------------ agent loop
def checkpoint(state):
    os.makedirs(STATE, exist_ok=True)
    tmp = os.path.join(STATE, f"{state['run_id']}.json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=1)
    os.replace(tmp, os.path.join(STATE, f"{state['run_id']}.json"))   # atomic: never a half-written state


def run_agent(run_id, customer_id, message, *, db_path=DB, token_budget=TOKEN_BUDGET, crash_after_tools=None,
              resume=False):
    path = os.path.join(STATE, f"{run_id}.json")
    if resume:
        with open(path, encoding="utf-8") as f:
            state = json.load(f)
        state["resumed"] = True
        print(f"  [resume] {run_id}: {state['steps']} steps and tools {state['tools_executed']} restored from {path}")
    else:
        state = {"run_id": run_id, "customer_id": customer_id, "message": message, "steps": 0, "tokens": 0,
                 "tools_executed": [], "status": "running", "resumed": False,
                 "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": message}]}
        checkpoint(state)
    tools = Tools(state["customer_id"], run_id, db_path)
    for m in state["messages"]:              # rebuild the verified-order cache from the restored transcript
        if m["role"] == "tool" and m["name"] == "get_order":
            o = json.loads(m["content"])
            if "error" not in o:
                tools.verified[o["order_id"]] = o

    final, outcome = None, None
    while True:
        if state["steps"] >= MAX_STEPS:
            outcome, final = "FALLBACK", (f"Handed to a human agent after {MAX_STEPS} steps. Summary: customer "
                                          f"{state['customer_id']} asked '{message}'; tools used: "
                                          f"{', '.join(state['tools_executed']) or 'none'}.")
            break
        if state["tokens"] >= token_budget:
            outcome, final = "STOPPED_BUDGET", (f"Stopped safely: token budget {token_budget} used up "
                                                f"({state['tokens']}); handed to a human agent.")
            break
        resp = mock_llm.chat(state["messages"], tools=TOOLS)
        state["steps"] += 1
        state["tokens"] += resp.usage.total_tokens
        msg = resp.choices[0].message
        if not msg.tool_calls:
            final = msg.content
            outcome = final.split(":", 1)[0].strip()
            break
        calls = [{"id": tc.id, "name": tc.name, "arguments": tc.arguments} for tc in msg.tool_calls]
        state["messages"].append({"role": "assistant", "content": None, "tool_calls": calls})
        for tc in calls:
            result = tools.execute(tc["name"], tc["arguments"])
            state["messages"].append({"role": "tool", "tool_call_id": tc["id"], "name": tc["name"],
                                      "content": json.dumps(result)})
            state["tools_executed"].append(tc["name"])
            checkpoint(state)
            # "order not found" is a business answer the model handles; anything else is a stop
            if "error" in result and (result.get("tool_failure") or tc["name"] != "get_order"):
                outcome, final = "STOPPED_TOOL_ERROR", f"Stopped safely: {tc['name']} failed ({result['error']})."
                break
            if crash_after_tools is not None and len(state["tools_executed"]) == crash_after_tools \
                    and not state["resumed"]:
                raise SimulatedCrash(f"process killed after {crash_after_tools} tool executions")
        if outcome:
            break

    refund = 0.0
    for m in state["messages"]:
        if m["role"] == "tool" and m["name"] in ("issue_refund", "escalate_to_human"):
            data = json.loads(m["content"])
            refund = data.get("refund_inr", refund)
            if data.get("status") == "already_refunded":     # trust the ledger, not the model's wording
                outcome, refund = "ALREADY_REFUNDED", 0.0
                final = f"No new payment: {data['refund_inr']} INR was already refunded for this order."
    state.update(status=outcome, final=final, refund_inr=refund)
    checkpoint(state)
    record = {k: state[k] for k in ("run_id", "customer_id", "message", "status", "refund_inr", "steps",
                                    "tokens", "tools_executed", "resumed", "final")}
    append_jsonl(RUNS, record)
    return record


# ------------------------------------------------------------------ demonstrations
def main():
    os.makedirs(OUT, exist_ok=True)
    for p in (RUNS, LEDGER):
        if os.path.exists(p):
            os.remove(p)
    shutil.rmtree(STATE, ignore_errors=True)
    os.makedirs(STATE)

    sc = pd.read_csv(os.path.join("data", "raw", "agent_scenarios.csv"))
    rows = []
    for s in sc.itertuples(index=False):
        r = run_agent(s.scenario_id, s.customer_id, s.message)
        ok = r["status"] == s.expected_outcome and abs(r["refund_inr"] - s.expected_refund_inr) < 0.005 \
            if s.expected_outcome in ("REFUND", "ESCALATE") else r["status"] == s.expected_outcome
        rows.append({"scenario": s.scenario_id, "customer": s.customer_id, "expected": s.expected_outcome,
                     "outcome": r["status"], "expected_inr": f"{s.expected_refund_inr:.2f}",
                     "refund_inr": f"{r['refund_inr']:.2f}", "steps": r["steps"], "tokens": r["tokens"],
                     "tools": " > ".join(r["tools_executed"]), "match": ok})
    res = pd.DataFrame(rows)
    print("=== seven scenarios ===")
    print(res.to_string(index=False))
    print(f"matched {int(res['match'].sum())}/{len(res)}")

    print("\n=== stop-safely demonstrations ===")
    s1 = sc.iloc[0]
    d1 = run_agent("DEMO_BUDGET", s1.customer_id, s1.message, token_budget=300)
    print(f"token budget 300 on {s1.scenario_id}: status {d1['status']} after {d1['steps']} steps, "
          f"{d1['tokens']} tokens, tools {d1['tools_executed']}")
    d2 = run_agent("DEMO_DB_DOWN", s1.customer_id, s1.message, db_path=os.path.join(STATE, "missing.db"))
    print(f"database unavailable on {s1.scenario_id}: status {d2['status']} after {d2['steps']} steps, "
          f"tools {d2['tools_executed']}")
    print(f"  final: {d2['final']}")
    d3 = run_agent("DEMO_REPEAT_S1", s1.customer_id, s1.message)
    paid = [e for e in read_ledger() if e["order_id"] == "ORD-100048" and e["action"] == "refund_issued"]
    print(f"(idempotency check) repeat of {s1.scenario_id}: status {d3['status']}; "
          f"ledger refund entries for that order: {len(paid)}")

    print("\n=== crash and resume ===")
    with sqlite3.connect(DB) as con:
        used = tuple(sc["message"].str.extract(r"(ORD-\d{6})")[0])
        oid, cid, amt = con.execute(
            "SELECT order_id, customer_id, order_amount FROM orders WHERE status='completed' AND delivery_minutes > 45 "
            f"AND order_amount <= 1000 AND order_id NOT IN ({','.join('?' * len(used))}) ORDER BY order_id LIMIT 1",
            used).fetchone()
    msg = f"My order {oid} reached me very late. I would like a refund."
    try:
        run_agent("DEMO_CRASH", cid, msg, crash_after_tools=2)
    except SimulatedCrash as e:
        with open(os.path.join(STATE, "DEMO_CRASH.json"), encoding="utf-8") as f:
            before = json.load(f)["tools_executed"]
        print(f"crash: {e}; tools executed before crash: {before}")
    r = run_agent("DEMO_CRASH", cid, None, resume=True)
    after = r["tools_executed"][len(before):]
    paid = [e for e in read_ledger() if e["order_id"] == oid and e["action"] == "refund_issued"]
    print(f"resumed: status {r['status']} INR {r['refund_inr']:.2f} (order amount INR {amt:.2f}); "
          f"tools executed after resume: {after}; full tool list: {r['tools_executed']}")
    print(f"ledger refund entries for {oid}: {len(paid)}")

    print(f"\nledger entries: {len(read_ledger())}; total refunded INR "
          f"{sum(e['refund_inr'] for e in read_ledger() if e['action'] == 'refund_issued'):.2f}")
    print(f"wrote {RUNS}, {LEDGER} and {STATE}/")


if __name__ == "__main__":
    main()
