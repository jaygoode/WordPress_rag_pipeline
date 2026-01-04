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




TOMORROW:
    -add logging OK
    -TESTS
    -write README
    -write improvements 
    -hook up the reranker
    -create the agent if time


