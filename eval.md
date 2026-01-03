3️⃣ Should you implement query and qrels usage?
    -✅ Yes, if your goal is to evaluate your retrieval pipeline.
    .Here’s how you could use them:
        .Load queries (query.json) → create Query objects.
        .Run retrieval → pass each query through your retriever (search).
        .Compare retrieved results against qrels.json → compute metrics:
            Recall@k → Did the relevant corpus document appear in the top-k results?
            MRR → Rank of first relevant document
            Precision → Fraction of top-k results that are relevant
        .Optional: Use queries and qrels to train or tune a reranker.

🔹 Recommended workflow
    -Keep your current ingestion pipeline as-is (corpus → embeddings → vector store).
    -Add an evaluation step using query.json + qrels.json:
    -Load queries
    -Retrieve top-k chunks
    -Check which chunks match qrels → compute metrics
    -This doesn’t change the ingestion pipeline, but gives you confidence your retriever works.
    -In short: query + qrels are for testing/evaluation, not for building the vector store.