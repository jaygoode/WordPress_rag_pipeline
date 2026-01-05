## Is the System “Good Enough”?

Before investing in further improvements, the key question is whether the system already meets a reasonable quality baseline.

### Manual Spot Check
Evaluate 10 random queries and inspect the top-5 retrieved results.

- **Are 2+ results relevant?** → ✅ Good  
- **Is the #1 result often relevant?** → ✅ Great  

### User Satisfaction
Assess whether users would realistically find the system helpful.

- **Can users find their answer in the top-5?** → ✅ Good enough  
- **Is the best answer usually ranked #1?** → ✅ Excellent  

### Business Impact
Determine whether the system effectively solves the underlying business problem.

- **Does it reduce support tickets?** → ✅ Success  
- **Do users prefer it over Google search?** → ✅ Excellent  

**Conclusion:**  
The system meets baseline expectations and justifies further iteration.

---

## Potential Improvements

### Infrastructure & Pipeline
- use openrouter api to easily switch between models and providers
- Introduce a test environment that automatically overwrites the database.
- for testing should also be able to limit the dataset for faster processes.
- Support dynamic embedding dimensions during ingestion.
- Improve observability by consistently collecting and analyzing run-level data.
- Avoid blind iteration; prioritize improvements based on measured impact.
- Consider tools such as **Langfuse** for tracing, evaluation, and analytics.

---

### Retrieval & Embeddings
- Normalize embeddings to unit length when using cosine similarity  
  (`embedding / ||embedding||`) to avoid scale-related degradation.
- Fine-tune embeddings on the target dataset (last resort if gains justify cost).
- Evaluate hybrid retrieval approaches:
  - **BM25 + vector embeddings**
  - Keyword matching can capture cases embeddings miss, especially in technical domains like WordPress.
- Tune chunking parameters:
  - Chunk size
  - Overlap
  - Maximum tokens
- Test additional embedding models within constraints:
  - Cost
  - Local vs. cloud execution
  - Licensing and legal considerations

---

### Reranker Observations
- Slight reduction in Recall@K observed.
- Marginal improvement in MRR.
- Higher-quality reranker models may yield better results but require testing.
- Potential improvement via fine-tuning on WordPress Q&A datasets.

---

### LLM Considerations (If Introduced)
Model selection should account for:

- Training data and domain alignment
- Legal compliance (open-source, open-weights, closed-source)
- Control and privacy requirements
- Customization capability
- Cost
- Output quality

#### Prompt Strategy
- Split a user query into multiple prompts for improved accuracy  
  (trade-off: increased token usage).

---

### LLM Evaluation Strategy
- Test **2–3 models**
- Use **20–50 representative prompts** per use case
- Score across:
  - Accuracy
  - Cost
  - Latency

If LLMs are integrated more deeply:

- Create **100–300 curated evaluation queries**
- Use:
  - Langfuse
  - BLEU / ROUGE
  - LLM-as-judge (e.g., GPT-5)

---

## Guiding Principle

Pipeline improvements should always be driven by data.  
By tracking performance and analyzing results, it becomes possible to identify where the largest gains are achievable—rather than iterating blindly.

