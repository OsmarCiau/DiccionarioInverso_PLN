import logging
import os
import random

from gensim.models import KeyedVectors
import networkx as nx
from node2vec import Node2Vec
import numpy as np

from reverse_dictionary.domain.models import (
    CandidateScore,
    Node2VecConfig,
)
from reverse_dictionary.domain.protocols import RetrievalStrategyProtocol

logger = logging.getLogger("embedding_adapters")


class Node2VecTrainer:
    def __init__(self, models_dir: str = "models") -> None:
        self._models_dir = models_dir
        os.makedirs(self._models_dir, exist_ok=True)

    def train_or_load(
        self, graph: nx.Graph, config: Node2VecConfig
    ) -> KeyedVectors:
        vectors_path = os.path.join(self._models_dir, f"{config.name}.kv")
        if os.path.exists(vectors_path):
            logger.info("Cargando modelo existente desde %s", vectors_path)
            return KeyedVectors.load(vectors_path)

        logger.info(
            "Iniciando entrenamiento de node2vec (%s): dim=%d, walk_len=%d, num_walks=%d, p=%.2f, q=%.2f",
            config.name,
            config.dimensions,
            config.walk_length,
            config.num_walks,
            config.p,
            config.q,
        )

        random.seed(config.seed)
        np.random.seed(config.seed)

        n2v = Node2Vec(
            graph,
            dimensions=config.dimensions,
            walk_length=config.walk_length,
            num_walks=config.num_walks,
            p=config.p,
            q=config.q,
            workers=1,
            seed=config.seed,
            quiet=True,
        )

        model = n2v.fit(
            window=5,
            min_count=1,
            batch_words=64,
            seed=config.seed,
            epochs=5,
        )

        keyed_vectors = model.wv
        keyed_vectors.save(vectors_path)
        logger.info("Modelo node2vec guardado en %s", vectors_path)

        return keyed_vectors


class Node2VecRetrievalStrategy(RetrievalStrategyProtocol):
    def __init__(
        self,
        strategy_name: str,
        keyed_vectors: KeyedVectors,
        target_vocabulary: set[str],
    ) -> None:
        self._strategy_name = strategy_name
        self._wv = keyed_vectors
        self._target_vocabulary = target_vocabulary

        self._candidate_words = [
            word for word in self._target_vocabulary if word in self._wv
        ]
        candidate_matrix = np.array(
            [self._wv[w] for w in self._candidate_words], dtype=np.float32
        )
        norms = np.linalg.norm(candidate_matrix, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        self._normalized_matrix = candidate_matrix / norms

    @property
    def strategy_name(self) -> str:
        return self._strategy_name

    def retrieve_candidates(
        self, query_lemmas: tuple[str, ...], top_k: int = 10
    ) -> tuple[CandidateScore, ...]:
        valid_query_vectors = [
            self._wv[lemma] for lemma in query_lemmas if lemma in self._wv
        ]
        if not valid_query_vectors:
            return ()

        query_composite = np.sum(valid_query_vectors, axis=0, dtype=np.float32)
        query_norm = np.linalg.norm(query_composite)
        if query_norm == 0.0:
            return ()
        query_unit = query_composite / query_norm

        similarities = np.dot(self._normalized_matrix, query_unit)
        query_lemmas_set = set(query_lemmas)

        scored_candidates: list[tuple[str, float]] = []
        for word, score in zip(self._candidate_words, similarities):
            if word not in query_lemmas_set:
                scored_candidates.append((word, float(score)))

        scored_candidates.sort(key=lambda item: item[1], reverse=True)

        results = [
            CandidateScore(word=word, score=round(score, 6), rank=idx + 1)
            for idx, (word, score) in enumerate(scored_candidates[:top_k])
        ]
        return tuple(results)


class PretrainedEmbeddingRetrievalStrategy(RetrievalStrategyProtocol):
    def __init__(
        self,
        strategy_name: str,
        keyed_vectors: KeyedVectors,
        target_vocabulary: set[str],
    ) -> None:
        self._strategy_name = strategy_name
        self._wv = keyed_vectors
        self._target_vocabulary = target_vocabulary

        self._candidate_words = [
            word for word in self._target_vocabulary if word in self._wv
        ]
        candidate_matrix = np.array(
            [self._wv[w] for w in self._candidate_words], dtype=np.float32
        )
        norms = np.linalg.norm(candidate_matrix, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        self._normalized_matrix = candidate_matrix / norms

    @property
    def strategy_name(self) -> str:
        return self._strategy_name

    @classmethod
    def load_from_gensim(
        cls,
        model_name: str = "glove-wiki-gigaword-50",
        target_vocabulary: set[str] | None = None,
    ) -> "PretrainedEmbeddingRetrievalStrategy":
        logger.info("Cargando modelo de embeddings preentrenado: %s", model_name)
        import gensim.downloader as api
        vectors = api.load(model_name)
        vocab = target_vocabulary if target_vocabulary is not None else set()
        return cls(strategy_name=f"baseline_{model_name}", keyed_vectors=vectors, target_vocabulary=vocab)

    def retrieve_candidates(
        self, query_lemmas: tuple[str, ...], top_k: int = 10
    ) -> tuple[CandidateScore, ...]:
        valid_query_vectors = [
            self._wv[lemma] for lemma in query_lemmas if lemma in self._wv
        ]
        if not valid_query_vectors:
            return ()

        query_composite = np.sum(valid_query_vectors, axis=0, dtype=np.float32)
        query_norm = np.linalg.norm(query_composite)
        if query_norm == 0.0:
            return ()
        query_unit = query_composite / query_norm

        similarities = np.dot(self._normalized_matrix, query_unit)
        query_lemmas_set = set(query_lemmas)

        scored_candidates: list[tuple[str, float]] = []
        for word, score in zip(self._candidate_words, similarities):
            if word not in query_lemmas_set:
                scored_candidates.append((word, float(score)))

        scored_candidates.sort(key=lambda item: item[1], reverse=True)

        results = [
            CandidateScore(word=word, score=round(score, 6), rank=idx + 1)
            for idx, (word, score) in enumerate(scored_candidates[:top_k])
        ]
        return tuple(results)


class ReciprocalRankFusionStrategy(RetrievalStrategyProtocol):
    def __init__(
        self,
        strategy_a: RetrievalStrategyProtocol,
        strategy_b: RetrievalStrategyProtocol,
        k_constant: int = 60,
    ) -> None:
        self._strategy_a = strategy_a
        self._strategy_b = strategy_b
        self._k_constant = k_constant

    @property
    def strategy_name(self) -> str:
        return f"fusion_rrf_{self._strategy_a.strategy_name}__and__{self._strategy_b.strategy_name}"

    def retrieve_candidates(
        self, query_lemmas: tuple[str, ...], top_k: int = 10
    ) -> tuple[CandidateScore, ...]:
        intermediate_k = max(top_k * 4, 40)
        results_a = self._strategy_a.retrieve_candidates(query_lemmas, top_k=intermediate_k)
        results_b = self._strategy_b.retrieve_candidates(query_lemmas, top_k=intermediate_k)

        rrf_scores: dict[str, float] = {}

        for item in results_a:
            rrf_scores[item.word] = rrf_scores.get(item.word, 0.0) + (1.0 / (self._k_constant + item.rank))

        for item in results_b:
            rrf_scores[item.word] = rrf_scores.get(item.word, 0.0) + (1.0 / (self._k_constant + item.rank))

        sorted_items = sorted(rrf_scores.items(), key=lambda item: item[1], reverse=True)

        fused_candidates = [
            CandidateScore(word=word, score=round(score, 6), rank=idx + 1)
            for idx, (word, score) in enumerate(sorted_items[:top_k])
        ]
        return tuple(fused_candidates)

