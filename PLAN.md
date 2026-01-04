corpus.json
{
    "_id":"21561",
    "title":"Where is the right place to register/enqueue scripts & styles",
    "text":"I am using WordPress 3.1.4 by now. I am confused with where (which hook) do I use:   * to register and/or enqueue    * scripts and styles    * on front- and back-ends? Questions:   * Which are right hooks to use?   * All front end register/enqueue scripts/styles in `init`?   * Why is there no `admin_print_styles-{xxx}`?"
}

qrels.json - Can be used to calculate metrics like Recall@k, Precision@k, or Mean Reciprocal Rank (MRR).
{
    "query-id":"120122",
    "corpus-id":"21561",
    "score":1.0
}

query.json
{
    "_id":"120122",
    "text":"How to enqueue script or style in a theme's template file?"
}

******************************CORE TASK***************************************
1. ingestion pipeline
    -cleans the raw data, OK   
    -chunks it appropriately, OK
    -persists it to the provided Postgres + pgvector instance (see docker-compose.yml) OK

2. retrieval system: 
    -embedding strategy, 
    -indexing approach, 
    -query handling

agent:
    BaseAgentController:
        -plan
        -serve
        -run

________________________________________________________________________

Embedding model: Which model you use (OpenAI, HuggingFace, LLaMA-based local models).
    Example free/local: sentence-transformers/all-MiniLM-L6-v2 (small, fast) or mistral-embedding on GPU.
    Example high-quality: OpenAI’s text-embedding-3-small or text-embedding-3-large.

Batching: Embed multiple chunks at once for efficiency (embed_batch).
Normalization: You might normalize vectors to unit length if your similarity metric is cosine similarity.

2. Indexing Approach
    USING hnsw (embedding vector_cosine_ops);


check: 

    nice to haves in future:
        -chunking strategies
        -embedding choices
        -check iter_queries in runner.py, should it be implemented in a real way?
        -WHY batching to DB?
        -USING hnsw (embedding vector_cosine_ops); WHY?
        -storage option other than default one
        -decide strategy for finding the stored embeddings, convert query to embedding with semantic similary search vs..
        -add llm to generate multiple queries from one, to increase search success
        -finetuning*
            -baseline b4 finetune -> really good training data, then finetune, then measure, ...'

    check before hand in:
        -search top k, rerank for top 5 , give answer
        -precision@k recall@k retrieval accuracy %
        -logging, tracking metrics - for prompts, rag, agents, overall system.
    

IF LLM
    100-300 test data sets 
        -langfuse, 
        -bleu, rouge
        -gpt5 llm as judge

create db volumes?


llm considerations: 
-legal compliance (open-source, open weights closed source)
-control and privacy
-customization
-cost 
-quality

test 2-3 models
    -20-50 prompts based on usecase 
    -score on accuracy cost speed etc

****************BENCHMARKS*******************************************
1. TREC (Text REtrieval Conference)
    The query–corpus–qrels paradigm you are using

MTEB (Massive Text Embedding Benchmark)
    A benchmark specifically for embeddings

study topics
    -TREC, MTEB, BEIR, MS MARCO
    -Appsettings, basesettings basemodel, schema.py pydantic_settings
    -extra keyword for logging, json output
    -NDCG (Normalized Discounted Cumulative Gain)


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

metrics
    pretrained SentenceTransformer: all-MiniLM-L6-v2
        -no reranker
        recall@5: 0.3579
        recall@10: 0.3579
        mrr: 0.2779


TOMORROW:
    -add logging OK
    -TESTS
    -write README
    -write improvements 
    -hook up the reranker
    -create the agent if time


TESTS:
storage:
    db connection
    db storage?
cli tests?

data/ingestion
    -chunk text
    -cleaning OK
    -rag_pipeline
    
embeddings - model
    embedding test?

eval
    -metrics 
        -recall@k 
        -mmr 
        -evaluate func
    -runner
        -QreelsEvaluator
            -load queries
            -load qrels
            -iter queries
            -evaluate

retrieval
   rerank
   retriever - search