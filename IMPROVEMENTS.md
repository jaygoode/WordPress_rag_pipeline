IMPROVEMENTS
    -test environment auto overwrites db 
    -dynamic embedding dimension
    -prompt to llm splits to multiple prompts for improved accuracy
    -Fine-tune embeddings for dataset
    -Normalize embeddings
        -Make sure all vectors are normalized to unit length if using cosine similarity (embedding / ||embedding||).
        -This prevents scale differences from reducing similarity scores.
    -Consider hybrid retrieval
        -Combine BM25 (keyword) + vector embeddings.
        -Sometimes exact keyword matches catch what embeddings miss, especially for technical Q&A like WordPress.
    -tweaking chunking params - overlap, max tokens

llm considerations: 
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