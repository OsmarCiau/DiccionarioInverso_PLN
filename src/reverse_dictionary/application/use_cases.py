import logging
from typing import Sequence

from reverse_dictionary.domain.models import (
    ErrorCaseRecord,
    EvaluationMetrics,
    EvaluationSample,
)
from reverse_dictionary.domain.protocols import ReverseDictionaryAgentProtocol

logger = logging.getLogger("use_cases")


class BenchmarkEvaluationUseCase:
    def __init__(self, agent: ReverseDictionaryAgentProtocol) -> None:
        self._agent = agent

    def evaluate_strategy(
        self,
        strategy_name: str,
        samples: Sequence[EvaluationSample],
        partition_label: str = "test",
        top_k: int = 10,
    ) -> EvaluationMetrics:
        total_samples = len(samples)
        if total_samples == 0:
            return EvaluationMetrics(
                strategy_name=strategy_name,
                partition=partition_label,
                precision_at_1=0.0,
                precision_at_3=0.0,
                precision_at_5=0.0,
                mean_reciprocal_rank=0.0,
                coverage_ratio=0.0,
                evaluated_definitions_count=0,
                average_execution_time_seconds=0.0,
            )

        hits_at_1 = 0
        hits_at_3 = 0
        hits_at_5 = 0
        sum_reciprocal_ranks = 0.0
        covered_count = 0
        total_time = 0.0

        for sample in samples:
            target_word = sample.concept.lower().strip()
            result = self._agent.query(sample.definition, strategy_name, top_k=top_k)
            total_time += result.execution_time_seconds

            if len(result.query_lemmas) > 0 and len(result.candidates) > 0:
                covered_count += 1

            candidate_words = [c.word.lower().strip() for c in result.candidates]

            target_rank = None
            if target_word in candidate_words:
                target_rank = candidate_words.index(target_word) + 1

            if target_rank is not None:
                sum_reciprocal_ranks += 1.0 / target_rank
                if target_rank == 1:
                    hits_at_1 += 1
                if target_rank <= 3:
                    hits_at_3 += 1
                if target_rank <= 5:
                    hits_at_5 += 1

        p_at_1 = round(hits_at_1 / total_samples, 4)
        p_at_3 = round(hits_at_3 / total_samples, 4)
        p_at_5 = round(hits_at_5 / total_samples, 4)
        mrr = round(sum_reciprocal_ranks / total_samples, 4)
        coverage = round(covered_count / total_samples, 4)
        avg_time = round(total_time / total_samples, 4)

        metrics = EvaluationMetrics(
            strategy_name=strategy_name,
            partition=partition_label,
            precision_at_1=p_at_1,
            precision_at_3=p_at_3,
            precision_at_5=p_at_5,
            mean_reciprocal_rank=mrr,
            coverage_ratio=coverage,
            evaluated_definitions_count=total_samples,
            average_execution_time_seconds=avg_time,
        )

        logger.info(
            "[%s | %s] p@1: %.3f | p@3: %.3f | p@5: %.3f | MRR: %.3f | Cobertura: %.3f | T_medio: %.4fs",
            strategy_name,
            partition_label,
            p_at_1,
            p_at_3,
            p_at_5,
            mrr,
            coverage,
            avg_time,
        )

        return metrics

    def evaluate_all_strategies(
        self,
        samples: Sequence[EvaluationSample],
        partition_label: str = "test",
        top_k: int = 10,
    ) -> tuple[EvaluationMetrics, ...]:
        metrics_list: list[EvaluationMetrics] = []
        for strat in self._agent.available_strategies:
            m = self.evaluate_strategy(strat, samples, partition_label=partition_label, top_k=top_k)
            metrics_list.append(m)
        return tuple(metrics_list)


class ErrorAnalysisUseCase:
    def __init__(self, agent: ReverseDictionaryAgentProtocol) -> None:
        self._agent = agent

    def extract_error_cases(
        self,
        strategy_name: str,
        samples: Sequence[EvaluationSample],
        max_cases: int = 15,
    ) -> tuple[ErrorCaseRecord, ...]:
        error_records: list[ErrorCaseRecord] = []

        for sample in samples:
            if len(error_records) >= max_cases:
                break

            target_word = sample.concept.lower().strip()
            result = self._agent.query(sample.definition, strategy_name, top_k=10)
            candidate_words = [c.word.lower().strip() for c in result.candidates]

            target_rank = None
            if target_word in candidate_words:
                target_rank = candidate_words.index(target_word) + 1

            if target_rank is None or target_rank > 5:
                top_5_words = tuple(candidate_words[:5])

                if len(result.query_lemmas) == 0:
                    category = "palabras_fuera_de_vocabulario"
                    explanation = "La definicion no genero ningun lema presente en el vocabulario del grafo USF."
                elif len(result.query_lemmas) <= 2:
                    category = "descripcion_ambigua_o_corta"
                    explanation = f"Pocos lemas utiles ({len(result.query_lemmas)}), lo que dispersa la centralidad asociativa."
                elif target_rank is not None:
                    category = "recuperacion_suboptima"
                    explanation = f"El concepto objetivo fue recuperado en la posicion {target_rank}, fuera del top 5."
                else:
                    category = "sesgo_asociativo_o_polisemia"
                    explanation = "Los lemas de la definicion evocaron asociaciones mas fuertes hacia otros conceptos semanticos."

                record = ErrorCaseRecord(
                    concept=target_word,
                    definition=sample.definition,
                    extracted_lemmas=result.query_lemmas,
                    predicted_top_5=top_5_words,
                    target_rank=target_rank,
                    error_category=category,
                    explanation=explanation,
                )
                error_records.append(record)

        logger.info(
            "Casos de error extraidos para %s: %d", strategy_name, len(error_records)
        )
        return tuple(error_records)

