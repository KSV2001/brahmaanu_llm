# brahmaanu_llm/rag/rag_config.py

# -------- Corpus I/O --------
RAG_DOCS_FOLDER = "data/raw/docs"          # folder with 8 .txt files
RAG_QUESTIONS_COL = "question"               # column name if using a DataFrame
RAG_PROMPTS_OUT_PARQUET = "eval/outputs/df_eval_rag_full_final_parquet" ## Already exists
RAG_INDEX_PATH = "rag/outputs/rag_index.pkl.gz"

# -------- Chunking --------
# options: "fact_mode" or "token_mode"
RAG_CHUNK_STRATEGY = "fact_mode"
RAG_N_UNITS = 1                               # Number of facts per chunk (fact_mode) or tokens per chunk (token_mode)

# Recognize lines like: [BHF-D021] some fact text...
RAG_CITATION_TAG_REGEX = r"\[(BHF-[A-Z]-?\d+)\]\s*(.*)"

# -------- Embeddings / Reranking --------
RAG_DENSE_MODEL = "BAAI/bge-large-en-v1.5"
RAG_EMBED_BATCH = 256
RAG_EMBED_NORMALIZE = True
RAG_CACHE_EMBEDS = True                       # saves .npy under docs folder

RAG_RERANK_MODEL = "BAAI/bge-reranker-base"
RAG_RERANK_BATCH_SIZE = 64

# -------- Retrieval knobs --------
RAG_BM25_TOPK = 200
RAG_DENSE_TOPK = 200
RAG_FUSED_LIMIT = 300                         # after RRF fusion
RAG_MMR_CANDIDATES = 80
RAG_MMR_LAMBDA = 0.7
RAG_TOPK = 10                                 # final context size

# RAG scoring weights
WEIGHT_CE_RANK = 0.60       # Cross-Encoder rank weight
WEIGHT_BM25 = 0.15          # BM25 (Okapi) best match weight
WEIGHT_TOKEN_OVERLAP = 0.05 # Token overlap weight
WEIGHT_BIGRAM_OVERLAP = 0.15 # Bi-gram overlap weight
WEIGHT_CLUSTER = 0.05       # Cluster match weight


# -------- Prompt text --------

RAG_PROMPT_RULES = (
    "You can use the provided CONTEXT to find the answers. "
    "If the CONTEXT contains the necessary data, you must find it to answer the question. "
    "The CONTEXT has fact_id, followed by the fact. "
    "You must return exactly the relevant fact_ids from the CONTEXT for the given question. "
    "Prioritize the schema of the returned JSON object."
)

RAG_PROMPT_JSON_RETURN_LINE = "Return JSON only. No markdown, no backticks, no extra keys."
