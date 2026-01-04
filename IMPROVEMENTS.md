IMPROVEMENTS
    -have test environment that auto overwrites db, and dynamic embedding dimension when ingesting 
    -user prompt to llm -> splits to multiple prompts for improved accuracy (downside token usage)
    -Fine-tune embeddings for dataset, finetune agent LLM ( if implemented, last resort? )
    -Normalize embeddings
        -Make sure all vectors are normalized to unit length if using cosine similarity (embedding / ||embedding||).
            -This prevents scale differences from reducing similarity scores.
    -Consider hybrid retrieval
        -Combine BM25 (keyword) + vector embeddings.
        -Sometimes exact keyword matches catch what embeddings miss, especially for technical Q&A like WordPress.
    -tweaking chunking params - overlap, max tokens
    -test more embeddings within restrictions(cost, local/cloud based, legality, and so on..)

llm considerations: 
    -choose models based on what they are trained on. 
    -legal compliance (open-source, open weights closed source)
    -control and privacy
    -customization
    -cost 
    -quality

test 2-3 models
    -20-50 prompts based on usecase 
    -score on accuracy cost speed etc

IF LLM
    100-300 test data sets 
        -langfuse, 
        -bleu, rouge
        -gpt5 llm as judge

for improving the pipeline we should always collect run data and be able to analyze it, to be able to improve the system in a way that makes sense, where can we get the biggest gains? 
    -dont want to go blindly without basing our next steps on data.



