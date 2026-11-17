import logging
from typing import Sequence

import spacy

from reverse_dictionary.domain.models import CorpusCoverageReport, EvaluationSample
from reverse_dictionary.domain.protocols import TextPreprocessorProtocol

logger = logging.getLogger("nlp")


class SpacyEnglishPreprocessor(TextPreprocessorProtocol):
    def __init__(self, model_name: str = "en_core_web_sm") -> None:
        logger.info("Loading spaCy language model: %s", model_name)
        self._nlp = spacy.load(model_name, disable=["ner"])

    def extract_lemmas(self, text: str) -> tuple[str, ...]:
        if not text or not text.strip():
            return ()

        doc = self._nlp(text)
        lemmas: list[str] = []

        for token in doc:
            if not token.is_alpha or token.is_stop or token.is_space:
                continue

            cleaned_lemma = token.lemma_.lower().strip()
            if len(cleaned_lemma) > 1:
                lemmas.append(cleaned_lemma)

        return tuple(lemmas)

    def extract_valid_query_lemmas(
        self, text: str, allowed_vocabulary: set[str]
    ) -> tuple[str, ...]:
        if not text or not text.strip():
            return ()

        doc = self._nlp(text)
        tokens_info = [
            (token.text.lower(), token.lemma_.lower(), token.is_stop)
            for token in doc
            if token.is_alpha and not token.is_space
        ]

        recognized_lemmas: list[str] = []
        skip_next = False

        for idx in range(len(tokens_info)):
            if skip_next:
                skip_next = False
                continue

            current_text, current_lemma, current_is_stop = tokens_info[idx]

            if idx + 1 < len(tokens_info):
                next_text, next_lemma, _ = tokens_info[idx + 1]
                bigram_text = f"{current_text}_{next_text}"
                bigram_lemma = f"{current_lemma}_{next_lemma}"

                if bigram_text in allowed_vocabulary:
                    recognized_lemmas.append(bigram_text)
                    skip_next = True
                    continue

                if bigram_lemma in allowed_vocabulary:
                    recognized_lemmas.append(bigram_lemma)
                    skip_next = True
                    continue

            if current_is_stop or len(current_lemma) <= 1:
                continue

            if current_lemma in allowed_vocabulary:
                recognized_lemmas.append(current_lemma)
            elif current_text in allowed_vocabulary:
                recognized_lemmas.append(current_text)

        unique_ordered: list[str] = []
        for lemma in recognized_lemmas:
            if lemma not in unique_ordered:
                unique_ordered.append(lemma)

        return tuple(unique_ordered)

    def calculate_corpus_coverage(
        self,
        samples: Sequence[EvaluationSample],
        allowed_vocabulary: set[str],
    ) -> CorpusCoverageReport:
        total = len(samples)
        if total == 0:
            return CorpusCoverageReport(
                total_definitions=0,
                definitions_with_graph_lemmas=0,
                coverage_percentage=0.0,
                average_useful_lemmas_per_definition=0.0,
            )

        with_useful_lemmas = 0
        total_useful_lemmas = 0

        for sample in samples:
            valid_lemmas = self.extract_valid_query_lemmas(sample.definition, allowed_vocabulary)
            lemmas_count = len(valid_lemmas)
            if lemmas_count > 0:
                with_useful_lemmas += 1
                total_useful_lemmas += lemmas_count

        coverage_pct = round((with_useful_lemmas / total) * 100.0, 2)
        avg_useful = round(total_useful_lemmas / total, 2)

        report = CorpusCoverageReport(
            total_definitions=total,
            definitions_with_graph_lemmas=with_useful_lemmas,
            coverage_percentage=coverage_pct,
            average_useful_lemmas_per_definition=avg_useful,
        )

        logger.info(
            "Cobertura de definiciones: %d/%d (%.2f%%), promedio de lemas utiles: %.2f",
            with_useful_lemmas,
            total,
            coverage_pct,
            avg_useful,
        )

        return report

