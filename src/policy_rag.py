"""Part D (tasks 1-4) -- policy RAG assistant: 40-word overlapping chunks, TF-IDF retrieval, grounded answers via
mock_llm POLICY ASSISTANT, and an evaluation on data/raw/policy_eval.csv with overlap 10 and overlap 0.
The retriever is reused by the refund agent's search_policy tool (src/part_d_agent.py).
Run from the repository root:  python src/policy_rag.py"""
import glob
import os
import sys

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import mock_llm  # noqa: E402

POL = os.path.join("data", "policies")
CHUNK_WORDS, OVERLAP = 40, 10
SYSTEM = ("You are the Instamart POLICY ASSISTANT. Answer only from the supplied CONTEXT. If the CONTEXT "
          "does not contain the answer, say: I could not find that information in the policy documents.")


def chunk_documents(folder=POL, size=CHUNK_WORDS, overlap=OVERLAP):
    """Chunks of `size` words, each starting size-overlap words after the previous one; stop after the first
    chunk whose end reaches or passes the last word of the document."""
    chunks = []
    for path in sorted(glob.glob(os.path.join(folder, "*.txt"))):
        words = open(path, encoding="utf-8").read().split()
        start = 0
        while True:
            chunks.append({"source": os.path.basename(path), "text": " ".join(words[start:start + size])})
            if start + size >= len(words):
                break
            start += size - overlap
    return chunks


class Retriever:
    def __init__(self, overlap=OVERLAP):
        self.chunks = chunk_documents(overlap=overlap)
        self.vectorizer = TfidfVectorizer()
        self.matrix = self.vectorizer.fit_transform([c["text"] for c in self.chunks])

    def retrieve(self, query, k=3):
        sims = cosine_similarity(self.vectorizer.transform([query]), self.matrix)[0]
        return [{"source": self.chunks[i]["source"], "score": round(float(sims[i]), 4),
                 "text": self.chunks[i]["text"]} for i in sims.argsort()[::-1][:k]]


_default = None


def retrieve(query, k=3):
    global _default
    _default = _default or Retriever()
    return _default.retrieve(query, k)


def answer(question, retriever=None):
    hits = retriever.retrieve(question, 3) if retriever else retrieve(question, 3)
    context = "\n".join(f"[{h['source']}] {h['text']}" for h in hits)
    messages = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": f"CONTEXT:\n{context}\nQUESTION:\n{question}"}]
    resp = mock_llm.chat(messages)
    return {"answer": resp.choices[0].message.content, "sources": [h["source"] for h in hits],
            "tokens": resp.usage.total_tokens}


def evaluate(overlap):
    r = Retriever(overlap=overlap)
    qa = pd.read_csv(os.path.join("data", "raw", "policy_eval.csv"), keep_default_na=False)
    rows = []
    for q in qa.itertuples(index=False):
        a = answer(q.question, r)
        answerable = q.expected_doc != "NONE"
        rows.append({"question": q.question, "answerable": answerable,
                     "doc_in_top3": (q.expected_doc in a["sources"]) if answerable else None,
                     "fact_in_answer": (q.expected_fact in a["answer"]) if answerable else None,
                     "fallback": a["answer"].startswith("I could not find"), "answer": a["answer"],
                     "tokens": a["tokens"]})
    return r, pd.DataFrame(rows)


def report(res, overlap, n_chunks):
    ans, un = res[res["answerable"]], res[~res["answerable"]]
    print(f"\n=== evaluation, overlap {overlap} ({n_chunks} chunks) ===")
    print(f"expected_doc in top 3: {int(ans['doc_in_top3'].sum())}/{len(ans)}; "
          f"expected_fact in answer: {int(ans['fact_in_answer'].sum())}/{len(ans)}; "
          f"fallback on unanswerable: {int(un['fallback'].sum())}/{len(un)}; tokens {int(res['tokens'].sum())}")
    for x in ans[~(ans["doc_in_top3"] & ans["fact_in_answer"])].itertuples():
        print(f"  FAIL: {x.question}\n        -> {x.answer}")


def main():
    r = Retriever()
    counts = pd.Series([c["source"] for c in r.chunks]).value_counts()
    print(f"chunks: {len(r.chunks)} -> {counts.to_dict()}")
    print("\nretrieve('refund policy late delivery'):")
    for h in r.retrieve("refund policy late delivery"):
        print(f"  {h['source']:22s} {h['score']:.4f}  {h['text'][:70]}...")
    for ov in (OVERLAP, 0):
        rr, res = evaluate(ov)
        report(res, ov, len(rr.chunks))
        if ov == OVERLAP:
            print(res[["question", "doc_in_top3", "fact_in_answer", "fallback"]].to_string(index=False))


if __name__ == "__main__":
    main()
