# brahmaanu_llm/rag/rag_pipeline.py
# Minimal dependencies: sentence-transformers, rank-bm25, faiss-cpu, regex, pandas (optional)

import os, re, glob
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd
import regex as rx
import faiss
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi
from collections import Counter
from brahmaanu_llm.configs.sft_config import SCHEMA_NOTE

from brahmaanu_llm.configs.rag_config import *

_TAG_RE = re.compile(RAG_CITATION_TAG_REGEX)

def _word_tokenize(s: str) -> List[str]:
    return rx.findall(r"\p{L}+\p{M}*|\d+(?:\.\d+)?|[^\s\p{L}\p{M}\d]", s.lower())

# ---------------- Chunker ----------------

class Chunker:
    """
    fact_mode: groups consecutive lines by N facts per chunk.
    token_mode: packs consecutive lines until >= N tokens per chunk.
    """
    def __init__(self, strategy: str, n_units: int):
        assert strategy in ("fact_mode", "token_mode")
        assert n_units >= 1
        self.strategy = strategy
        self.n_units = n_units

    def parse_doc_lines(self, path: str) -> List[Tuple[str, str]]:
        """
        Returns list of (tag, text) from a .txt doc.
        Lines must be like: [BHF-...] body...
        """
        items = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                m = _TAG_RE.match(line)
                if not m:
                    continue
                tag, body = m.group(1), m.group(2)
                items.append((tag, body))
        return items

    def chunk_doc(self, doc_id: str, path: str) -> List[Dict[str, Any]]:
        facts = self.parse_doc_lines(path)
        chunks: List[Dict[str, Any]] = []
        if not facts:
            return chunks

        if self.strategy == "fact_mode":
            # group every n_units facts
            for i in range(0, len(facts), self.n_units):
                group = facts[i:i+self.n_units]
                tags = [t for t, _ in group]
                body = "\n".join(f"[{t}] {b}" for t, b in group)
                chunks.append({"doc_id": doc_id, "fact_tags": tags, "text": body})

        else:  # token_mode
            cur_tags: List[str] = []
            cur_lines: List[str] = []
            cur_tok = 0
            for t, b in facts:
                line = f"[{t}] {b}"
                ltoks = len(_word_tokenize(line))
                if cur_tok == 0:
                    cur_tags, cur_lines, cur_tok = [t], [line], ltoks
                else:
                    if cur_tok + ltoks > self.n_units and cur_lines:
                        # flush current
                        chunks.append({
                            "doc_id": doc_id,
                            "fact_tags": cur_tags,
                            "text": "\n".join(cur_lines),
                        })
                        cur_tags, cur_lines, cur_tok = [t], [line], ltoks
                    else:
                        cur_tags.append(t)
                        cur_lines.append(line)
                        cur_tok += ltoks
            # flush last
            if cur_lines:
                chunks.append({"doc_id": doc_id, "fact_tags": cur_tags, "text": "\n".join(cur_lines)})

        return chunks

# ---------------- RAG Index ----------------

class RAGIndex:
    def __init__(
        self,
        folder: str = RAG_DOCS_FOLDER,
        dense_model: str = RAG_DENSE_MODEL,
        reranker_model: str = RAG_RERANK_MODEL,
        rerank_batch_size: int = RAG_RERANK_BATCH_SIZE,
        bm25_topk: int = RAG_BM25_TOPK,
        dense_topk: int = RAG_DENSE_TOPK,
        fused_limit: int = RAG_FUSED_LIMIT,
        mmr_candidates: int = RAG_MMR_CANDIDATES,
        mmr_lambda: float = RAG_MMR_LAMBDA,
        chunk_strategy: str = RAG_CHUNK_STRATEGY,
        n_units: int = RAG_N_UNITS,
        embed_batch: int = RAG_EMBED_BATCH,
        embed_normalize: bool = RAG_EMBED_NORMALIZE,
        cache_embeds: bool = RAG_CACHE_EMBEDS,
    ):
        self.folder = folder
        self.items: List[Dict[str, Any]] = []
        self.texts: List[str] = []
        self.doc_ids: List[str] = []
        self.tags_list: List[List[str]] = []

        self.model_dense = SentenceTransformer(dense_model)
        self.reranker = CrossEncoder(reranker_model)
        self.rerank_batch_size = rerank_batch_size

        self.bm25_topk = bm25_topk
        self.dense_topk = dense_topk
        self.fused_limit = fused_limit
        self.mmr_candidates = mmr_candidates
        self.mmr_lambda = mmr_lambda

        self.embed_batch = embed_batch
        self.embed_normalize = embed_normalize
        self.cache_embeds = cache_embeds

        self.chunker = Chunker(chunk_strategy, n_units)

        self.index: faiss.Index = None  # type: ignore
        self.vecs: Optional[np.ndarray] = None
        self.bm25: Optional[BM25Okapi] = None

        self._load_docs()
        self._build()

    def _load_docs(self):
        for path in sorted(glob.glob(os.path.join(self.folder, "*.txt"))):
            doc_id = os.path.splitext(os.path.basename(path))[0]
            chunks = self.chunker.chunk_doc(doc_id, path)
            for ch in chunks:
                self.items.append(ch)
                self.texts.append(ch["text"])
                self.doc_ids.append(ch["doc_id"])
                self.tags_list.append(ch["fact_tags"])

    def _build(self):
        if not self.texts:
            dim = self.model_dense.get_sentence_embedding_dimension()
            self.index = faiss.IndexFlatIP(dim)
            self.vecs = np.zeros((0, dim), dtype="float32")
            self.bm25 = BM25Okapi([])
            return

        dim = self.model_dense.get_sentence_embedding_dimension()
        cache = os.path.join(
            self.folder,
            f"emb_{dim}_{len(self.texts)}_{int(self.embed_normalize)}_{self.chunker.strategy}_{self.chunker.n_units}.npy"
        )

        if self.cache_embeds and os.path.exists(cache):
            X = np.load(cache).astype("float32")
        else:
            X = self.model_dense.encode(
                self.texts,
                batch_size=self.embed_batch,
                normalize_embeddings=self.embed_normalize,
                show_progress_bar=False,
            ).astype("float32")
            if self.cache_embeds:
                np.save(cache, X)

        self.index = faiss.IndexFlatIP(X.shape[1])
        self.index.add(X)
        self.vecs = X

        tokenized = [_word_tokenize(t) for t in self.texts]
        self.bm25 = BM25Okapi(tokenized)

    # ---- search helpers ----
    @staticmethod
    def _multi_queries(q: str) -> List[str]:
        q = q.strip()
        q_lower = q.lower()
        q_nopunct = "".join(ch for ch in q if ch.isalnum() or ch.isspace())
        seen = set(); out = []
        for s in (q, q_lower, q_nopunct):
            if s and s not in seen:
                out.append(s); seen.add(s)
        return out

    def _dense_search(self, q: str, topk: int) -> List[int]:
        qv = self.model_dense.encode([q], normalize_embeddings=self.embed_normalize, show_progress_bar=False).astype("float32")
        D, I = self.index.search(qv, topk)
        return I[0].tolist()

    def _bm25_search(self, q: str, topk: int) -> List[int]:
        toks = _word_tokenize(q)
        scores = self.bm25.get_scores(toks)  # type: ignore
        order = np.argsort(-scores)[:topk]
        return order.tolist()

    @staticmethod
    def _rrf(ranklists: List[List[int]], k: int = 60, c: int = 60, limit: int = 100) -> List[int]:
        score: Dict[int, float] = {}
        for lst in ranklists:
            for r, idx in enumerate(lst[:k]):
                score[idx] = score.get(idx, 0.0) + 1.0 / (c + r + 1)
        return [i for i, _ in sorted(score.items(), key=lambda x: -x[1])][:limit]

    # ---- re-score (lexical + bm25 + doc cluster) ----
    def _bigrams(self, toks: List[str]):
        return set(zip(toks, toks[1:])) if len(toks) > 1 else set()

    def _bm25_scores_for_ids(self, q: str, ids: List[int]) -> List[float]:
        toks = _word_tokenize(q)
        all_scores = self.bm25.get_scores(toks)  # type: ignore
        return [float(all_scores[i]) for i in ids]

    def _rescore(self, question: str, cand_ids: List[int], ce_scores: List[float]) -> List[Tuple[int, float]]:
        q_toks = _word_tokenize(question)
        q_set, q_bi = set(q_toks), self._bigrams(q_toks)
        txts = [self.texts[i] for i in cand_ids]
        doc_toks = [_word_tokenize(t) for t in txts]
        tok_overlap = [len(q_set.intersection(set(t))) / (len(q_set) or 1) for t in doc_toks]
        bi_overlap = [len(q_bi.intersection(self._bigrams(t))) / (len(q_bi) or 1) for t in doc_toks]

        bm25_raw = self._bm25_scores_for_ids(question, cand_ids)
        bmin, bmax = min(bm25_raw), max(bm25_raw)
        bm25_norm = [(s - bmin) / ((bmax - bmin) or 1.0) for s in bm25_raw]

        
        docs = [self.items[i]["doc_id"] for i in cand_ids]
        top_docs = {d for d, _ in Counter(docs).most_common(3)}
        cluster = [1.0 if d in top_docs else 0.0 for d in docs]


        # Calculate final scores
        final = [
            WEIGHT_CE_RANK * float(ce) + 
            WEIGHT_BM25 * bm + 
            WEIGHT_TOKEN_OVERLAP * tok + 
            WEIGHT_BIGRAM_OVERLAP * bi + 
            WEIGHT_CLUSTER * cl
            for ce, bm, tok, bi, cl in zip(ce_scores, bm25_norm, tok_overlap, bi_overlap, cluster)
        ]
        ranked = sorted(zip(cand_ids, final), key=lambda x: -x[1])
        return ranked

    # ---- MMR ----
    def _mmr(self, cand_ids: List[int], cand_scores: List[float], k: int, lam: float) -> List[int]:
        if not cand_ids:
            return []
        selected: List[int] = []
        vec = self.vecs  # type: ignore
        while cand_ids and len(selected) < k:
            best_i, best_val = -1, -1e9
            for i, (cid, s) in enumerate(zip(cand_ids, cand_scores)):
                rep = 0.0
                if selected:
                    rep = max(float(np.dot(vec[cid], vec[sid])) for sid in selected)
                val = lam * s - (1.0 - lam) * rep
                if val > best_val:
                    best_val, best_i = val, i
            chosen = cand_ids.pop(best_i)
            cand_scores.pop(best_i)
            selected.append(chosen)
        return selected

    # ---- Public retrieve ----
    def retrieve(self, question: str, topk: int = RAG_TOPK) -> List[Dict[str, Any]]:
        if not self.texts:
            return []
        queries = self._multi_queries(question)
        dense_lists = [self._dense_search(q, topk=self.dense_topk) for q in queries]
        bm25_lists = [self._bm25_search(q, topk=self.bm25_topk) for q in queries]
        fused = self._rrf(dense_lists + bm25_lists, limit=self.fused_limit)

        pairs = [[question, self.texts[i]] for i in fused]
        ce_scores: List[float] = []
        for i in range(0, len(pairs), self.rerank_batch_size):
            batch_scores = self.reranker.predict(pairs[i:i + self.rerank_batch_size])
            ce_scores.extend([float(s) for s in np.asarray(batch_scores).ravel().tolist()])

        ranked_ce = sorted(zip(fused, ce_scores), key=lambda x: -x[1])
        cand_ids = [i for i, _ in ranked_ce[:self.mmr_candidates]]
        cand_scores = [s for _, s in ranked_ce[:self.mmr_candidates]]

        ranked_mix = self._rescore(question, cand_ids, cand_scores)
        cand_ids = [i for i, _ in ranked_mix]
        cand_scores = [s for _, s in ranked_mix]

        kept = self._mmr(cand_ids, cand_scores, k=topk, lam=self.mmr_lambda)
        score_map = dict(ranked_mix)

        out = []
        for i in kept:
            rec = self.items[i]
            out.append({
                "doc_id": rec["doc_id"],
                "fact_tags": rec["fact_tags"],
                "text": rec["text"],
                "score": float(score_map.get(i, 0.0)),
            })
        return out

# ---------------- Prompting ----------------

def build_prompt(question: str, ctx: Optional[List[Dict[str, Any]]] = None) -> str:
    if ctx and len(ctx) > 0:
        ctx_str = "\n".join(c["text"] for c in ctx)
        body = f"{SCHEMA_NOTE}\n\n{RAG_PROMPT_RULES}\n\nQuestion:\n{question}\n\nCONTEXT:\n{ctx_str}"
    else:
        body = f"{SCHEMA_NOTE}\n\nQUESTION:\n{question}"
    return f"<s>[INST] {body}\n\n{RAG_PROMPT_JSON_RETURN_LINE} [/INST]\n"

# ---------------- Driver helpers ----------------

def build_index(folder_with_docs: str = RAG_DOCS_FOLDER, **overrides) -> RAGIndex:
    return RAGIndex(folder=folder_with_docs, **overrides)

def create_rag_prompts(
    questions: List[str],
    index: Optional[RAGIndex] = None,
    topk: int = RAG_TOPK,
    return_ctx: bool = False,
) -> Tuple[List[str], Optional[List[List[Dict[str, Any]]]]]:
    idx = index or build_index()
    prompts: List[str] = []
    ctxs_all: List[List[Dict[str, Any]]] = []
    for q in questions:
        ctx = idx.retrieve(q, topk=topk)
        rag_prompt = build_prompt(q, ctx)
        prompts.append(rag_prompt)
        ctxs_all.append(ctx)
    return (prompts, ctxs_all) if return_ctx else (prompts, None)

def create_rag_prompts_from_df(
    df_questions: pd.DataFrame,
    question_col: str = RAG_QUESTIONS_COL,
    out_parquet: Optional[str] = RAG_PROMPTS_OUT_PARQUET,
    index: Optional[RAGIndex] = None,
    topk: int = RAG_TOPK,
) -> pd.DataFrame:
    qs = df_questions[question_col].astype(str).tolist()
    prompts, _ = create_rag_prompts(qs, index=index, topk=topk, return_ctx=False)
    df_out = df_questions.copy()
    df_out["rag_prompt"] = prompts
    if out_parquet:
        os.makedirs(os.path.dirname(out_parquet), exist_ok=True)
        df_out.to_parquet(out_parquet, index=False)
    return df_out

if __name__ == "__main__":
    import os, pickle, argparse
    import pandas as pd
    from brahmaanu_llm.rag.rag_pipeline import build_index, create_rag_prompts_from_df
    from brahmaanu_llm.rag.rag_config import (
        RAG_DOCS_FOLDER, RAG_QUESTIONS_COL, RAG_PROMPTS_OUT_PARQUET
    )

    ap = argparse.ArgumentParser()
    ap.add_argument("--df", required=True, help="Questions file (.parquet or .csv)")
    ap.add_argument("--index-pkl", default=None, help="Optional path to a pickled RAGIndex")
    ap.add_argument("--docs-folder", default=RAG_DOCS_FOLDER, help="Folder with .txt fact files")
    ap.add_argument("--question-col", default=RAG_QUESTIONS_COL, help="Column name with questions")
    ap.add_argument("--out-parquet", default=RAG_PROMPTS_OUT_PARQUET, help="Output parquet with rag_prompt")
    args = ap.parse_args()

    # Load questions DF
    df = pd.read_parquet(args.df) if args.df.endswith(".parquet") else pd.read_csv(args.df)

    # Load or build index
    idx = None
    if args.index_pkl and os.path.exists(args.index_pkl):
        try:
            with open(args.index_pkl, "rb") as f:
                idx = pickle.load(f)
        except Exception:
            idx = build_index(folder_with_docs=args.docs_folder)
    else:
        idx = build_index(folder_with_docs=args.docs_folder)

    # Generate RAG prompts and save
    df_out = create_rag_prompts_from_df(
        df_questions=df,
        question_col=args.question_col,
        out_parquet=args.out_parquet,
        index=idx,
    )

    # Optional: persist the index for faster reuse
    if args.index_pkl:
        try:
            with open(args.index_pkl, "wb") as f:
                pickle.dump(idx, f)
        except Exception:
            pass
