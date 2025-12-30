# prepare_medpsych_for_dar.py
import json
from pathlib import Path

from datasets import load_dataset  # pip install datasets

DATASET_NAME = "169Pi/medical_psychology"
MAX_EXAMPLES = 20000  # keep this manageable; bump if you want

def main():
    out_dir = Path("paper_experiments/data/medpsych_online")
    out_dir.mkdir(parents=True, exist_ok=True)

    ds = load_dataset(DATASET_NAME, split="train")
    print(f"Loaded {len(ds)} examples from {DATASET_NAME}")

    # Optional: sub-sample for speed
    if MAX_EXAMPLES and len(ds) > MAX_EXAMPLES:
        ds = ds.select(range(MAX_EXAMPLES))
        print(f"Subsampled to {len(ds)} examples")

    corpus_path = out_dir / "corpus.jsonl"
    queries_path = out_dir / "queries.jsonl"
    qrels_path = out_dir / "qrels.jsonl"

    with corpus_path.open("w", encoding="utf-8") as f_corpus, \
         queries_path.open("w", encoding="utf-8") as f_queries, \
         qrels_path.open("w", encoding="utf-8") as f_qrels:
        for i, row in enumerate(ds):
            doc_id = f"d{i}"
            q_id = f"q{i}"

            prompt = (row.get("prompt") or "").strip()
            cot = (row.get("complex_cot") or "").strip()
            answer = (row.get("response") or "").strip()

            # document text = reasoning + final answer
            doc_text_parts = [p for p in [cot, answer] if p]
            if not doc_text_parts:
                # skip pathological rows
                continue
            doc_text = "\n\n".join(doc_text_parts)

            # corpus entry
            corpus_obj = {
                "id": doc_id,
                "title": prompt[:120],  # short “title” from the prompt
                "text": doc_text,
            }
            f_corpus.write(json.dumps(corpus_obj) + "\n")

            # query entry
            query_obj = {
                "id": q_id,
                "text": prompt,
            }
            f_queries.write(json.dumps(query_obj) + "\n")

            # qrels: each query relevant to its own document
            qrels_obj = {
                "query-id": q_id,
                "corpus-id": doc_id,
                "score": 1,
            }
            f_qrels.write(json.dumps(qrels_obj) + "\n")

    print("Wrote:")
    print(f"  {corpus_path}")
    print(f"  {queries_path}")
    print(f"  {qrels_path}")
    print("Now call dar_router_main.py pointing to this workdir.")

if __name__ == "__main__":
    main()
