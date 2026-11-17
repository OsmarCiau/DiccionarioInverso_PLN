from typing import Literal, Protocol

from reverse_dictionary.domain.models import (
    CandidateScore,
    EvaluationSample,
    GraphSummary,
    LexicalQueryResult,
    WordAssociationEdge,
)


class TextPreprocessorProtocol(Protocol):
    def extract_lemmas(self, text: str) -> tuple[str, ...]:
        ...

    def extract_valid_query_lemmas(
        self, text: str, allowed_vocabulary: set[str]
    ) -> tuple[str, ...]:
        ...


class AssociationNormsLoaderProtocol(Protocol):
    def load_edges(self) -> tuple[WordAssociationEdge, ...]:
        ...

    def load_vocabulary(self) -> set[str]:
        ...


class EvaluationCorpusLoaderProtocol(Protocol):
    def load_samples(
        self, partition: Literal["dev", "test"] | None = None
    ) -> tuple[EvaluationSample, ...]:
        ...


class GraphAnalyzerProtocol(Protocol):
    def get_summary(self) -> GraphSummary:
        ...

    def export_degree_distribution_plot(self, output_image_path: str) -> None:
        ...


class RetrievalStrategyProtocol(Protocol):
    @property
    def strategy_name(self) -> str:
        ...

    def retrieve_candidates(
        self, query_lemmas: tuple[str, ...], top_k: int = 10
    ) -> tuple[CandidateScore, ...]:
        ...


class ReverseDictionaryAgentProtocol(Protocol):
    @property
    def available_strategies(self) -> tuple[str, ...]:
        ...

    def query(
        self, definition: str, strategy_name: str, top_k: int = 10
    ) -> LexicalQueryResult:
        ...

