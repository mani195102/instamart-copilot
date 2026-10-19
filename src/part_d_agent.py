"""Part D (tasks 5-8) -- refund agent: TOOLS_SPEC + TOOL_MAP over data/instamart.db and the policy retriever,
a ReAct loop with iteration / token / time limits, checkpoints in state/ and resume_agent().
Run from the repository root (after Part A):  python src/part_d_agent.py"""
import json
import os
import shutil
import sqlite3
import sys
import time
from datetime import datetime

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mock_llm  # noqa: E402
from policy_rag import retrieve  # noqa: E402

DB = os.path.join("data", "instamart.db")
OUT, STATE = "outputs", "state"
RUNS, LEDGER = os.path.join(OUT, "agent_runs.jsonl"), os.path.join(OUT, "refund_ledger.jsonl")
LATE_MINUTES, AUTO_LIMIT_INR = 45, 1000
SYSTEM = ("You are the Instamart refund AGENT. Use the tools to look up the order, check the refund policy, "
          "compute the refund, then issue it or escalate it to a human. Never refund more than the policy allows.")


# ------------------------------------------------------------------ tools
def _ledger(tool, order_id, refund_inr, run_id=None):
    with open(LEDGER, "a", encoding="utf-8") as f:
        f.write(json.dumps({"timestamp": datetime.now().isoformat(timespec="seconds"), "tool": tool,
                            "order_id": order_id, "refund_inr": refund_inr, "run_id": run_id}) + "\n")


def _already_issued(order_id):
    if not os.path.exists(LEDGER):
        return False
    with open(LEDGER, encoding="utf-8") as f:
        return any(json.loads(l)["tool"] == "issue_refund" and json.loads(l)["order_id"] == order_id for l in f)


def get_order(order_id):
    with sqlite3.connect(DB) as con:
        row = con.execute("SELECT order_id, status, order_amount, delivery_minutes FROM orders WHERE order_id = ?",
                          (order_id,)).fetchone()
    if row is None:
        return {"error": "order not found"}
    return dict(zip(["order_id", "status", "order_amount", "delivery_minutes"], row))


def search_policy(query):
    return {"results": retrieve(query, k=3)}


def calculate_refund(order_amount, delivery_minutes):
    late = delivery_minutes is not None and float(delivery_minutes) > LATE_MINUTES
    return {"refund_inr": round(float(order_amount), 2) if late else 0.0}


def issue_refund(order_id, refund_inr):
    if float(refund_inr) > AUTO_LIMIT_INR:                       # guardrail: never auto-pay above the limit
        return {"error": f"refund above INR {AUTO_LIMIT_INR} must be escalated", "refund_inr": 0.0}
    if _already_issued(order_id):                                # idempotency: never pay one order twice
        return {"refund_inr": refund_inr, "note": "already refunded earlier; no second payment"}
    _ledger("issue_refund", order_id, refund_inr, _current_run)
    return {"refund_inr": refund_inr}


def escalate_to_human(order_id, refund_inr):
    _ledger("escalate_to_human", order_id, refund_inr, _current_run)
    return {"refund_inr": refund_inr}


TOOLS_SPEC = [
    {"name": "get_order", "description": "Look up one order in the Instamart database by its order id.",
     "parameters": {"type": "object", "properties": {"order_id": {"type": "string"}}, "required": ["order_id"]}},
    {"name": "search_policy", "description": "Retrieve the three policy chunks most relevant to a query.",
     "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}},
    {"name": "calculate_refund",
     "description": "Compute the refund: full order amount if delivered in more than 45 minutes, else 0.",
     "parameters": {"type": "object", "properties": {"order_amount": {"type": "number"},
                                                     "delivery_minutes": {"type": "number"}},
                    "required": ["order_amount", "delivery_minutes"]}},
    {"name": "issue_refund", "description": "Pay an approved refund of at most INR 1000 and record it in the ledger.",
     "parameters": {"type": "object", "properties": {"order_id": {"type": "string"}, "refund_inr": {"type": "number"}},
                    "required": ["order_id", "refund_inr"]}},
    {"name": "escalate_to_human", "description": "Send a refund above INR 1000 to a human support lead for approval.",
     "parameters": {"type": "object", "properties": {"order_id": {"type": "string"}, "refund_inr": {"type": "number"}},
                    "required": ["order_id", "refund_inr"]}},
]
TOOL_MAP = {"get_order": get_order, "search_policy": search_policy, "calculate_refund": calculate_refund,
            "issue_refund": issue_refund, "escalate_to_human": escalate_to_human}
_current_run = None


# ------------------------------------------------------------------ agent loop
class SimulatedCrash(Exception):
    """Injected process crash for the crash-and-resume demonstration."""


def _save(state):
    os.makedirs(STATE, exist_ok=True)
    tmp = os.path.join(STATE, f"{state['run_id']}.json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=1)
    target = os.path.join(STATE, f"{state['run_id']}.json")
    for attempt in range(20):            # atomic replace; retry while a sync client (e.g. OneDrive) holds the file
        try:
            os.replace(tmp, target)
            return
        except PermissionError:
            if attempt == 19:
                raise
            time.sleep(0.1)


def _handoff(state, reason):
    tools = ", ".join(state["tools_executed"]) or "none"
    return (f"Handed to a human agent ({reason}). Request: '{state['messages'][1]['content']}'. "
            f"Tools completed: {tools}. Model calls: {state['llm_calls']}, tokens: {state['tokens_used']}.")


def _loop(state, crash_after_tools=None):
    global _current_run
    _current_run = state["run_id"]
    started = time.monotonic()
    status = final = None
    while True:
        if time.monotonic() - started >= state["time_limit_seconds"]:
            status, final = "FALLBACK_TIMEOUT", _handoff(state, "time limit reached")
            break
        if state["llm_calls"] >= state["max_iterations"]:
            status, final = "FALLBACK_MAX_ITERATIONS", _handoff(state, f"{state['max_iterations']} model calls "
                                                                       "without a final answer")
            break
        resp = mock_llm.chat(state["messages"], tools=TOOLS_SPEC)
        state["llm_calls"] += 1
        state["tokens_used"] += resp.usage.total_tokens
        msg = resp.choices[0].message
        if state["tokens_used"] > state["token_limit"]:          # stop before executing any requested tool
            status, final = "FALLBACK_TOKEN_LIMIT", _handoff(state, f"token limit {state['token_limit']} exceeded")
            break
        if not msg.tool_calls:
            final = msg.content
            status = final.split(":", 1)[0].strip()
            break
        calls = [{"id": tc.id, "name": tc.name, "arguments": tc.arguments} for tc in msg.tool_calls]
        state["messages"].append({"role": "assistant", "content": None, "tool_calls": calls})
        for tc in calls:
            fn = TOOL_MAP.get(tc["name"])
            obs = fn(**tc["arguments"]) if fn else {"error": f"unknown tool {tc['name']}"}
            state["messages"].append({"role": "tool", "tool_call_id": tc["id"], "name": tc["name"],
                                      "content": json.dumps(obs)})
            state["step"] += 1
            state["tools_executed"].append(tc["name"])
            state["steps"].append({"step": state["step"], "tool": tc["name"], "arguments": tc["arguments"],
                                   "observation": obs})
            _save(state)
            if crash_after_tools is not None and len(state["tools_executed"]) == crash_after_tools:
                raise SimulatedCrash(f"simulated crash after tool execution {crash_after_tools}")

    refund = 0.0
    if status in ("REFUND", "ESCALATE"):
        refund = next(s["observation"]["refund_inr"] for s in reversed(state["steps"])
                      if s["tool"] in ("issue_refund", "escalate_to_human"))
    state.update(status=status, final_answer=final, refund_inr=refund, finished=True)
    _save(state)
    with open(RUNS, "a", encoding="utf-8") as f:
        f.write(json.dumps({k: state[k] for k in ("run_id", "scenario_id", "status", "final_answer", "refund_inr",
                                                  "llm_calls", "tokens_used", "resumed", "steps")}) + "\n")
    return {k: state[k] for k in ("status", "final_answer", "steps", "llm_calls", "tokens_used", "refund_inr",
                                  "tools_executed")}


def run_agent(user_message, scenario_id, max_iterations=6, token_limit=6000, time_limit_seconds=10,
              run_id=None, crash_after_tools=None):
    state = {"run_id": run_id or scenario_id, "scenario_id": scenario_id, "max_iterations": max_iterations,
             "token_limit": token_limit, "time_limit_seconds": time_limit_seconds, "llm_calls": 0,
             "tokens_used": 0, "step": 0, "tools_executed": [], "steps": [], "resumed": False, "finished": False,
             "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user_message}]}
    return _loop(state, crash_after_tools)


def resume_agent(run_id):
    with open(os.path.join(STATE, f"{run_id}.json"), encoding="utf-8") as f:
        state = json.load(f)
    if state.get("finished"):
        raise ValueError(f"run {run_id} already finished")
    state["resumed"] = True
    return _loop(state)


# ------------------------------------------------------------------ demonstrations
def main():
    os.makedirs(OUT, exist_ok=True)
    for p in (RUNS, LEDGER):
        if os.path.exists(p):
            os.remove(p)
    shutil.rmtree(STATE, ignore_errors=True)

    sc = pd.read_csv(os.path.join("data", "raw", "agent_scenarios.csv"))
    rows = []
    for s in sc.itertuples(index=False):
        r = run_agent(s.message, s.scenario_id)
        expected = "FALLBACK_MAX_ITERATIONS" if s.expected_outcome == "FALLBACK" else s.expected_outcome
        ok = r["status"] == expected and abs(r["refund_inr"] - s.expected_refund_inr) < 0.01
        rows.append({"scenario": s.scenario_id, "expected": s.expected_outcome, "status": r["status"],
                     "refund_inr": f"{r['refund_inr']:.2f}", "llm_calls": r["llm_calls"],
                     "tokens_used": r["tokens_used"], "tools": " > ".join(r["tools_executed"]), "match": ok})
    res = pd.DataFrame(rows)
    print("=== seven scenarios (max_iterations=6, token_limit=6000, time_limit_seconds=10) ===")
    print(res.to_string(index=False))
    print(f"matched {int(res['match'].sum())}/{len(res)}")
    assert res["match"].all()

    print("\n=== stop-safely demonstrations ===")
    s1 = sc.iloc[0]
    d = run_agent(s1.message, "S1", token_limit=200, run_id="S1_token_limit_200")
    print(f"S1 token_limit=200: {d['status']} after {d['llm_calls']} model calls, {d['tokens_used']} tokens, "
          f"tools {d['tools_executed']}\n  {d['final_answer']}")
    d = run_agent(s1.message, "S1", time_limit_seconds=0, run_id="S1_time_limit_0")
    print(f"S1 time_limit_seconds=0: {d['status']} after {d['llm_calls']} model calls\n  {d['final_answer']}")

    print("\n=== crash and resume (S1) ===")
    try:
        run_agent(s1.message, "S1", run_id="S1_crash_resume", crash_after_tools=2)
    except SimulatedCrash as e:
        with open(os.path.join(STATE, "S1_crash_resume.json"), encoding="utf-8") as f:
            before = json.load(f)["tools_executed"]
        print(f"{e}; checkpoint holds tools {before}")
    r = resume_agent("S1_crash_resume")
    print(f"resumed: {r['status']} INR {r['refund_inr']:.2f}; tools across crash + resume: {r['tools_executed']}")
    assert r["tools_executed"] == ["get_order", "search_policy", "calculate_refund", "issue_refund"]
    assert r["status"] == "REFUND" and abs(r["refund_inr"] - 128.25) < 0.01

    with open(LEDGER, encoding="utf-8") as f:
        ledger = [json.loads(l) for l in f]
    print(f"\nledger: {[(e['tool'], e['order_id'], e['refund_inr']) for e in ledger]}")
    print(f"wrote {RUNS}, {LEDGER} and {STATE}/")


if __name__ == "__main__":
    main()
