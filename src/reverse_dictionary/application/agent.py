import logging
import time

from reverse_dictionary.domain.models import (
    EmptyQueryError,
    InvalidWeightSchemeError,
    LexicalQueryResult,
)
from reverse_dictionary.domain.protocols import (
    RetrievalStrategyProtocol,
    ReverseDictionaryAgentProtocol,
    TextPreprocessorProtocol,
)

logger = logging.getLogger("agent")


class ReverseDictionaryAgent(ReverseDictionaryAgentProtocol):
    def __init__(
        self,
        preprocessor: TextPreprocessorProtocol,
        target_vocabulary: set[str],
        strategies: dict[str, RetrievalStrategyProtocol] | None = None,
    ) -> None:
        self._preprocessor = preprocessor
        self._target_vocabulary = target_vocabulary
        self._strategies: dict[str, RetrievalStrategyProtocol] = strategies or {}

    @property
    def available_strategies(self) -> tuple[str, ...]:
        return tuple(self._strategies.keys())

    def register_strategy(self, strategy: RetrievalStrategyProtocol) -> None:
        name = strategy.strategy_name
        self._strategies[name] = strategy
        logger.info("Estrategia registrada: %s", name)

    def query(
        self, definition: str, strategy_name: str, top_k: int = 10
    ) -> LexicalQueryResult:
        if not definition or not definition.strip():
            raise EmptyQueryError("La definicion de entrada no puede estar vacia")

        if strategy_name not in self._strategies:
            raise InvalidWeightSchemeError(
                f"Estrategia desconocida: '{strategy_name}'. Disponibles: {self.available_strategies}"
            )

        start_time = time.perf_counter()
        strategy = self._strategies[strategy_name]

        query_lemmas = self._preprocessor.extract_valid_query_lemmas(
            definition, self._target_vocabulary
        )

        if not query_lemmas:
            elapsed = time.perf_counter() - start_time
            return LexicalQueryResult(
                query=definition,
                query_lemmas=(),
                candidates=(),
                execution_time_seconds=round(elapsed, 4),
                strategy_used=strategy_name,
            )

        candidates = strategy.retrieve_candidates(query_lemmas, top_k=top_k)
        elapsed = time.perf_counter() - start_time

        return LexicalQueryResult(
            query=definition,
            query_lemmas=query_lemmas,
            candidates=candidates,
            execution_time_seconds=round(elapsed, 4),
            strategy_used=strategy_name,
        )

