import logging
from typing import Literal

import pandas as pd

from reverse_dictionary.domain.models import EvaluationSample, WordAssociationEdge
from reverse_dictionary.domain.protocols import (
    AssociationNormsLoaderProtocol,
    EvaluationCorpusLoaderProtocol,
)

logger = logging.getLogger("loaders")


class CsvAssociationNormsLoader(AssociationNormsLoaderProtocol):
    def __init__(self, csv_path: str = "data/usf_norms.csv") -> None:
        self._csv_path = csv_path

    def load_edges(self) -> tuple[WordAssociationEdge, ...]:
        logger.info("Cargando normas asociativas desde %s", self._csv_path)
        df = pd.read_csv(self._csv_path, keep_default_na=False)

        edges: list[WordAssociationEdge] = []
        for row in df.itertuples(index=False):
            source = str(row.estimulo).strip()
            target = str(row.respuesta).strip()
            if not source or not target:
                continue

            backward_str = row.fuerza_inversa
            try:
                backward_val = float(backward_str) if backward_str != "" else None
            except (ValueError, TypeError):
                backward_val = None

            edge = WordAssociationEdge(
                source=source,
                target=target,
                frequency=int(row.frecuencia),
                forward_strength=float(row.fuerza),
                backward_strength=backward_val,
                distance_frequency=float(row.dist_frecuencia),
                distance_strength=float(row.dist_fuerza),
            )
            edges.append(edge)

        logger.info("Total de aristas cargadas: %d", len(edges))
        return tuple(edges)

    def load_vocabulary(self) -> set[str]:
        df = pd.read_csv(self._csv_path, keep_default_na=False)
        cues = set(str(item).strip() for item in df["estimulo"] if str(item).strip())
        targets = set(str(item).strip() for item in df["respuesta"] if str(item).strip())
        vocab = cues.union(targets)
        logger.info("Vocabulario total cargado: %d palabras", len(vocab))
        return vocab


class CsvEvaluationCorpusLoader(EvaluationCorpusLoaderProtocol):
    def __init__(self, csv_path: str = "data/evaluation_corpus.csv") -> None:
        self._csv_path = csv_path

    def load_samples(
        self, partition: Literal["dev", "test"] | None = None
    ) -> tuple[EvaluationSample, ...]:
        logger.info("Cargando corpus de evaluacion desde %s (particion: %s)", self._csv_path, partition)
        df = pd.read_csv(self._csv_path, keep_default_na=False)

        if partition is not None:
            df = df[df["particion"] == partition]

        samples: list[EvaluationSample] = []
        for row in df.itertuples(index=False):
            sample = EvaluationSample(
                concept=str(row.concepto).strip(),
                definition=str(row.definicion).strip(),
                source=str(row.fuente).strip(),
                partition=row.particion,
            )
            samples.append(sample)

        logger.info("Muestras de evaluacion cargadas: %d", len(samples))
        return tuple(samples)

