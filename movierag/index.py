"""Step 2: indexing and retrieval.

Two complementary ways to find candidates for a query:

* Vector search (semantic): embed the query, find the nearest title vectors.
  Good at "a slow, eerie film about loneliness" -> Under the Skin.
* BM25 (lexical): classic keyword scoring, as in search engines.
  Good at names and rare words: "Shaolin", "Paganini", "Nuremberg".

Hybrid search runs both and merges the two rankings with Reciprocal Rank
Fusion (RRF). One document per title: our texts are short, so no chunking.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Iterable, Optional

from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.schema import NodeWithScore, TextNode
from llama_index.core.vector_stores import FilterOperator, MetadataFilter, MetadataFilters

from .schemas import Title

log = logging.getLogger(__name__)


def title_to_node(t: Title) -> TextNode:
    meta = {
        "item_id": t.item_id,
        "title": t.title,
        "kind": t.kind,
        "source": t.source,
        "year": t.year,
        "runtime_min": t.runtime_min,
        "genres": "|".join(t.genres),
    }
    meta = {k: v for k, v in meta.items() if v not in (None, "")}  # Chroma rejects None
    node = TextNode(id_=t.item_id, text=t.embedding_text(), metadata=meta)
    # By default LlamaIndex prepends metadata to the text it embeds / shows the LLM.
    # Our text already contains everything, so switch that off.
    node.excluded_embed_metadata_keys = list(meta)
    node.excluded_llm_metadata_keys = list(meta)
    return node


def rrf_fuse(rankings: Iterable[list[NodeWithScore]], k: int = 60, top_k: int | None = None) -> list[NodeWithScore]:
    """Reciprocal Rank Fusion: score(d) = sum over rankings of 1 / (k + rank_d).

    Uses only *ranks*, never raw scores, so it can merge a cosine similarity
    (0..1) with a BM25 score (0..30) without any calibration. k=60 is the value
    from the original paper (Cormack et al., 2009); it damps the top ranks.
    """
    scores: dict[str, float] = {}
    nodes: dict[str, NodeWithScore] = {}
    for ranking in rankings:
        for rank, nws in enumerate(ranking, start=1):
            nid = nws.node.node_id
            scores[nid] = scores.get(nid, 0.0) + 1.0 / (k + rank)
            nodes.setdefault(nid, nws)
    fused = sorted(scores, key=scores.get, reverse=True)[:top_k]
    return [NodeWithScore(node=nodes[n].node, score=scores[n]) for n in fused]


def _fingerprint(nodes: list[TextNode], embed_name: str) -> str:
    h = hashlib.sha1(embed_name.encode())
    for n in nodes:
        h.update(n.node_id.encode())
        h.update(n.text.encode())
    return h.hexdigest()[:16]


class MovieIndex:
    """Vector index (Chroma) + BM25 over the same nodes."""

    def __init__(
        self,
        titles: list[Title],
        embed_model,
        persist_dir: Optional[Path] = None,
        collection: str = "my_list",
        rebuild: bool = False,
    ):
        self.titles = titles
        self.by_id = {t.item_id: t for t in titles}
        self.embed_model = embed_model
        self.nodes = [title_to_node(t) for t in titles]
        self.node_by_id = {n.node_id: n for n in self.nodes}
        self._bm25_all = None
        self.index = self._build_vector_index(persist_dir, collection, rebuild)

    # ---------------------------------------------------------------- build
    def _build_vector_index(self, persist_dir, collection, rebuild) -> VectorStoreIndex:
        import chromadb
        from llama_index.vector_stores.chroma import ChromaVectorStore

        embed_name = getattr(self.embed_model, "model_name", type(self.embed_model).__name__)
        fp = _fingerprint(self.nodes, embed_name)

        if persist_dir is None:
            client = chromadb.EphemeralClient()
        else:
            client = chromadb.PersistentClient(path=str(persist_dir))

        existing = {c.name if hasattr(c, "name") else c for c in client.list_collections()}
        if collection in existing:
            coll = client.get_collection(collection)
            same = (coll.metadata or {}).get("fingerprint") == fp and coll.count() == len(self.nodes)
            if same and not rebuild:
                log.info("Re-using persisted Chroma collection %s", collection)
                store = ChromaVectorStore(chroma_collection=coll)
                return VectorStoreIndex.from_vector_store(store, embed_model=self.embed_model)
            client.delete_collection(collection)  # stale (catalogue or embedder changed)

        coll = client.create_collection(
            collection,
            metadata={"hnsw:space": "cosine", "fingerprint": fp, "embed_model": embed_name},
        )
        store = ChromaVectorStore(chroma_collection=coll)
        storage = StorageContext.from_defaults(vector_store=store)
        return VectorStoreIndex(self.nodes, storage_context=storage, embed_model=self.embed_model)

    # ---------------------------------------------------------------- search
    def vector_search(self, query: str, top_k: int = 10, allowed_ids: Optional[set[str]] = None) -> list[NodeWithScore]:
        if allowed_ids is not None and not allowed_ids:
            return []
        filters = None
        if allowed_ids is not None and len(allowed_ids) < len(self.nodes):
            filters = MetadataFilters(
                filters=[MetadataFilter(key="item_id", value=sorted(allowed_ids), operator=FilterOperator.IN)]
            )
        k = min(top_k, len(allowed_ids) if allowed_ids is not None else len(self.nodes))
        retriever = self.index.as_retriever(similarity_top_k=k, filters=filters)
        return retriever.retrieve(query)

    def bm25_search(self, query: str, top_k: int = 10, allowed_ids: Optional[set[str]] = None) -> list[NodeWithScore]:
        from llama_index.retrievers.bm25 import BM25Retriever

        if allowed_ids is None:
            nodes = self.nodes
        else:
            nodes = [self.node_by_id[i] for i in allowed_ids if i in self.node_by_id]
        if not nodes:
            return []
        k = min(top_k, len(nodes))
        if allowed_ids is None:
            if self._bm25_all is None:
                self._bm25_all = BM25Retriever.from_defaults(nodes=self.nodes, similarity_top_k=len(self.nodes))
            retriever = self._bm25_all
        else:  # a few dozen docs: building a filtered BM25 index takes milliseconds
            retriever = BM25Retriever.from_defaults(nodes=nodes, similarity_top_k=k)
        results = retriever.retrieve(query)
        return [r for r in results if (r.score or 0) > 0][:k]  # drop zero-overlap docs

    def hybrid_search(self, query: str, top_k: int = 10, allowed_ids: Optional[set[str]] = None) -> list[NodeWithScore]:
        pool = max(top_k * 3, 20)
        return rrf_fuse(
            [self.vector_search(query, pool, allowed_ids), self.bm25_search(query, pool, allowed_ids)],
            top_k=top_k,
        )

    def search(self, query: str, method: str = "hybrid", top_k: int = 10, allowed_ids=None, reranker=None):
        fn = {"vector": self.vector_search, "bm25": self.bm25_search, "hybrid": self.hybrid_search}[method]
        hits = fn(query, top_k=max(top_k, 15) if reranker else top_k, allowed_ids=allowed_ids)
        if reranker is not None and hits:
            hits = rerank(reranker, query, hits, top_k)
        return hits[:top_k]

    def titles_for(self, hits: list[NodeWithScore]) -> list[tuple[Title, float]]:
        return [(self.by_id[h.node.node_id], float(h.score or 0)) for h in hits if h.node.node_id in self.by_id]


def rerank(reranker, query: str, hits: list[NodeWithScore], top_n: int) -> list[NodeWithScore]:
    """Cross-encoder pass: reads (query, document) *together* -> sharper ordering."""
    from llama_index.core.schema import QueryBundle

    reranker.top_n = top_n
    return reranker.postprocess_nodes(hits, query_bundle=QueryBundle(query))


def rank_in_memory(titles: list[Title], query: str, embed_model, top_k: int = 10) -> list[NodeWithScore]:
    """Rank a small, temporary set (e.g. TMDb discoveries) with a throwaway index.

    No Chroma here: LlamaIndex's default in-memory SimpleVectorStore is plenty
    for 30 items, which also shows that the vector store is just a plug-in.
    """
    if not titles:
        return []
    nodes = [title_to_node(t) for t in titles]
    index = VectorStoreIndex(nodes, embed_model=embed_model)
    return index.as_retriever(similarity_top_k=min(top_k, len(nodes))).retrieve(query)
