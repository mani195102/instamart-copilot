"""Part D -- policy RAG assistant: sentence chunks, TF-IDF retrieval, grounded answers with citations,
a refusal guard for out-of-scope questions, and an evaluation on data/raw/policy_eval.csv.
The retriever is reused by the refund agent's search_policy tool (src/part_d_agent.py).
Run from the repository root:  python src/policy_rag.py"""
import glob
import os
import re
import sys

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import mock_llm  # noqa: E402

POL = os.path.join("data", "policies")
TOP_K = 3
MIN_SCORE = 0.10            # below this best-chunk similarity the question is out of scope
REFUSAL = "I could not find that information in the policy documents."
SYSTEM = ("You are the Instamart POLICY ASSISTANT. Answer only from the CONTEXT. "
          "If the CONTEXT does not contain the answer, say you could not find it. Cite the source file.")


class PolicyIndex:
    def __init__(self, folder=POL):
        self.chunks = []                          # (doc, sentence)
        for path in sorted(glob.glob(os.path.join(folder, "*.txt"))):
            text = open(path, encoding="utf-8").read().strip()
            for s in re.split(r"(?<=[.])\s+", text):
                if s.strip():
                    self.chunks.append((os.path.basename(path), s.strip()))
        self.vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True)
        self.mat = self.vec.fit_transform([s for _, s in self.chunks])

    def retrieve(self, query, k=TOP_K):
        sims = cosine_similarity(self.vec.transform([query]), self.mat)[0]
        order = sims.argsort()[::-1][:k]
        return [{"doc": self.chunks[i][0], "text": self.chunks[i][1], "score": round(float(sims[i]), 4)}
                for i in order]


def answer(index, question):
    hits = index.retrieve(question)
    if not hits or hits[0]["score"] < MIN_SCORE:
        return {"answer": REFUSAL, "sources": [], "hits": hits, "tokens": 0, "refused": True}
    context = "\n".join(f"[{h['doc']}] {h['text']}" for h in hits)
    messages = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": f"CONTEXT:\n{context}\nQUESTION: {question}"}]
    resp = mock_llm.chat(messages)
    text = resp.choices[0].message.content
    source = next((h["doc"] for h in hits if text.strip() in h["text"] or h["text"] in text), None)
    return {"answer": text, "sources": [source] if source else [], "hits": hits,
            "tokens": resp.usage.total_tokens, "refused": text == REFUSAL}


def evaluate():
    index = PolicyIndex()
    print(f"policy chunks: {len(index.chunks)} sentences from {len({d for d, _ in index.chunks})} files")
    qa = pd.read_csv(os.path.join("data", "raw", "policy_eval.csv"), keep_default_na=False)
    rows = []
    for q in qa.itertuples(index=False):
        r = answer(index, q.question)
        in_scope = q.expected_doc != "NONE"
        rows.append({
            "question": q.question, "expected_doc": q.expected_doc,
            "top_doc": r["hits"][0]["doc"], "top_score": r["hits"][0]["score"],
            "retrieval_hit": (q.expected_doc in {h["doc"] for h in r["hits"]}) if in_scope else None,
            "correct": (q.expected_fact.lower() in r["answer"].lower()) if in_scope else r["refused"],
            "cited": r["sources"][0] if r["sources"] else "", "tokens": r["tokens"], "answer": r["answer"]})
    res = pd.DataFrame(rows)
    ins = res[res["expected_doc"] != "NONE"]
    oos = res[res["expected_doc"] == "NONE"]
    print(res[["question", "top_doc", "top_score", "retrieval_hit", "correct", "tokens"]].to_string(index=False))
    print(f"\nretrieval hit@{TOP_K}: {int(ins['retrieval_hit'].sum())}/{len(ins)}; "
          f"top-1 doc correct: {int((ins['top_doc'] == ins['expected_doc']).sum())}/{len(ins)}")
    print(f"answer contains expected fact: {int(ins['correct'].sum())}/{len(ins)}; "
          f"out-of-scope refused: {int(oos['correct'].sum())}/{len(oos)}; total tokens {int(res['tokens'].sum())}")
    for r in res[~res["correct"].astype(bool)].itertuples():
        print(f"MISS: {r.question} -> {r.answer}")
    return res


if __name__ == "__main__":
    evaluate()
