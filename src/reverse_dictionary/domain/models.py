from dataclasses import dataclass
from typing import Literal


class ReverseDictionaryDomainError(Exception):
    pass


class LexicalItemNotFoundError(ReverseDictionaryDomainError):
    pass


class EmptyQueryError(ReverseDictionaryDomainError):
    pass


class InvalidWeightSchemeError(ReverseDictionaryDomainError):
    pass


WeightScheme = Literal["frequency", "strength"]


@dataclass(frozen=True)
class WordAssociationEdge:
    source: str
    target: str
    frequency: int
    forward_strength: float
    backward_strength: float | None
    distance_frequency: float
    distance_strength: float


@dataclass(frozen=True)
class CandidateScore:
    word: str
    score: float
    rank: int


@dataclass(frozen=True)
class EvaluationSample:
    concept: str
    definition: str
    source: str
    partition: Literal["dev", "test"]


@dataclass(frozen=True)
class LexicalQueryResult:
    query: str
    query_lemmas: tuple[str, ...]
    candidates: tuple[CandidateScore, ...]
    execution_time_seconds: float
    strategy_used: str


@dataclass(frozen=True)
class EvaluationMetrics:
    strategy_name: str
    partition: str
    precision_at_1: float
    precision_at_3: float
    precision_at_5: float
    mean_reciprocal_rank: float
    coverage_ratio: float
    evaluated_definitions_count: int
    average_execution_time_seconds: float


@dataclass(frozen=True)
class GraphSummary:
    node_count: int
    edge_count: int
    average_degree: float
    connected_components_count: int
    largest_component_size: int
    average_clustering_coefficient: float


@dataclass(frozen=True)
class CorpusCoverageReport:
    total_definitions: int
    definitions_with_graph_lemmas: int
    coverage_percentage: float
    average_useful_lemmas_per_definition: float


@dataclass(frozen=True)
class ErrorCaseRecord:
    concept: str
    definition: str
    extracted_lemmas: tuple[str, ...]
    predicted_top_5: tuple[str, ...]
    target_rank: int | None
    error_category: str
    explanation: str


@dataclass(frozen=True)
class Node2VecConfig:
    name: str
    dimensions: int
    walk_length: int
    num_walks: int
    p: float
    q: float
    seed: int = 42
