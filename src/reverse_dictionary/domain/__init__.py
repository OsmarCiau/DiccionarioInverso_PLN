"""Modulo de dominio para el sistema de Diccionario Inverso."""

from reverse_dictionary.domain.models import (
    CandidateScore,
    CorpusCoverageReport,
    EmptyQueryError,
    ErrorCaseRecord,
    EvaluationMetrics,
    EvaluationSample,
    GraphSummary,
    InvalidWeightSchemeError,
    LexicalItemNotFoundError,
    LexicalQueryResult,
    Node2VecConfig,
    ReverseDictionaryDomainError,
    WeightScheme,
    WordAssociationEdge,
)
from reverse_dictionary.domain.protocols import (
    AssociationNormsLoaderProtocol,
    EvaluationCorpusLoaderProtocol,
    GraphAnalyzerProtocol,
    RetrievalStrategyProtocol,
    ReverseDictionaryAgentProtocol,
    TextPreprocessorProtocol,
)

__all__ = [
    "AssociationNormsLoaderProtocol",
    "CandidateScore",
    "CorpusCoverageReport",
    "EmptyQueryError",
    "ErrorCaseRecord",
    "EvaluationCorpusLoaderProtocol",
    "EvaluationMetrics",
    "EvaluationSample",
    "GraphAnalyzerProtocol",
    "GraphSummary",
    "InvalidWeightSchemeError",
    "LexicalItemNotFoundError",
    "LexicalQueryResult",
    "Node2VecConfig",
    "RetrievalStrategyProtocol",
    "ReverseDictionaryAgentProtocol",
    "ReverseDictionaryDomainError",
    "TextPreprocessorProtocol",
    "WeightScheme",
    "WordAssociationEdge",
]

