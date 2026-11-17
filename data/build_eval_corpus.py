import argparse
import logging
import random
import re
from typing import Any

from nltk.corpus import wordnet as wn
from nltk.stem import PorterStemmer
import pandas as pd
import spacy

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("build_eval_corpus")

stemmer = PorterStemmer()


def is_valid_definition(concept: str, definition: str, content_lemmas: list[str]) -> bool:
    if len(definition.split()) < 5:
        return False

    concept_normalized = concept.lower().replace("_", " ")
    definition_lower = definition.lower()
    concept_stem = stemmer.stem(concept_normalized)

    escaped_pattern = r"\b" + re.escape(concept_normalized) + r"\b"
    if re.search(escaped_pattern, definition_lower):
        return False

    concept_tokens = concept_normalized.split()
    for token in concept_tokens:
        if len(token) > 3 and token in content_lemmas:
            return False

    definition_words = re.findall(r"[a-zA-Z]+", definition_lower)
    for word in definition_words:
        if len(word) >= 3 and stemmer.stem(word) == concept_stem:
            return False
        if len(concept_normalized) >= 4 and concept_normalized in word:
            return False

    return True


def collect_evaluation_samples(
    usf_csv_path: str,
    target_sample_size: int,
    dev_size: int,
    random_seed: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    usf_df = pd.read_csv(usf_csv_path, keep_default_na=False)
    graph_vocab = set(str(w) for w in usf_df["estimulo"]).union(set(str(w) for w in usf_df["respuesta"]))
    cues_pool = sorted(set(str(w) for w in usf_df["estimulo"].unique()))

    nlp = spacy.load("en_core_web_sm", disable=["ner"])

    rng = random.Random(random_seed)
    shuffled_cues = list(cues_pool)
    rng.shuffle(shuffled_cues)

    selected_records: list[dict[str, str]] = []

    for cue in shuffled_cues:
        if len(selected_records) >= target_sample_size:
            break

        if len(cue) < 3 or "_" in cue:
            continue

        synsets = wn.synsets(cue)
        if not synsets:
            continue

        chosen_definition = ""
        for synset in synsets:
            candidate_def = synset.definition().strip()
            doc = nlp(candidate_def)
            content_lemmas = [
                token.lemma_.lower()
                for token in doc
                if token.is_alpha and not token.is_stop and len(token.lemma_) > 2
            ]

            lemmas_in_graph = [lemma for lemma in content_lemmas if lemma in graph_vocab]

            if len(lemmas_in_graph) >= 2 and is_valid_definition(cue, candidate_def, content_lemmas):
                chosen_definition = candidate_def
                break

        if chosen_definition:
            selected_records.append({
                "concepto": cue,
                "definicion": chosen_definition,
                "fuente": "wordnet",
            })

    if len(selected_records) < target_sample_size:
        raise ValueError(
            f"No se alcanzaron las {target_sample_size} definiciones validas. Se obtuvieron {len(selected_records)}."
        )

    rng.shuffle(selected_records)

    for idx, record in enumerate(selected_records):
        record["particion"] = "dev" if idx < dev_size else "test"

    df_eval = pd.DataFrame(selected_records)

    dev_count = int((df_eval["particion"] == "dev").sum())
    test_count = int((df_eval["particion"] == "test").sum())

    stats = {
        "total_definitions": len(df_eval),
        "dev_definitions": dev_count,
        "test_definitions": test_count,
        "source": "WordNet (nltk.corpus.wordnet)",
    }

    return df_eval, stats


def update_data_dictionary(dictionary_path: str, eval_stats: dict[str, Any]) -> None:
    appendix = f"""

---

## 4. Conjunto de Evaluacion: Definiciones en Lenguaje Natural

- **Archivo:** `data/evaluation_corpus.csv`
- **Fuente primaria:** {eval_stats['source']}.
- **Total de definiciones evaluadas:** {eval_stats['total_definitions']:,}
- **Particion de desarrollo (`dev`):** {eval_stats['dev_definitions']:,} definiciones (destinadas a la exploracion y calibracion de hiperparametros).
- **Particion de prueba (`test`):** {eval_stats['test_definitions']:,} definiciones (destinadas a la evaluacion comparativa final e informe de resultados).

### Descripcion de Campos del Corpus de Evaluacion

| Campo | Tipo de Dato | Valores / Rango | Descripcion |
|---|---|---|---|
| `concepto` | `str` | Lema en minusculas | Palabra objetivo (*target concept*) que debe ser recuperada por el diccionario inverso. Existe como nodo activo dentro del grafo USF. |
| `definicion` | `str` | Texto en lenguaje natural | Descripcion o glosa en ingles que describe el concepto sin contener el lema objetivo. |
| `fuente` | `str` | `wordnet` | Recurso lexicografico de procedencia de la definicion. |
| `particion` | `str` | `dev` / `test` | Segmento asignado de forma determinista (`seed=42`) para la experimentacion. |

### Criterios de Aceptacion y Control de Fugas
1. **Existencia en el Grafo:** Todo `concepto` corresponde a una palabra de estimulo normada con vecindad estructurada en USF.
2. **Inclusion de Lemas Utiles:** Cada definicion posee al menos dos lemas lexicos con representacion en el vocabulario de USF.
3. **Ausencia Estricta de la Palabra Objetivo:** Se valido mediante analisis lexico y lematizacion/stemming (PorterStemmer) que ni el lema, ni su raiz morfologica, ni subcadenas aparezcan dentro de la definicion, evitando atajos o trampas de evaluacion.
"""
    with open(dictionary_path, "a", encoding="utf-8") as fp:
        fp.write(appendix)

    logger.info("Diccionario de datos actualizado en: %s", dictionary_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Construir corpus de evaluacion para el Diccionario Inverso.")
    parser.add_argument("--usf_csv", type=str, default="data/usf_norms.csv", help="Ruta de las normas procesadas USF")
    parser.add_argument("--output_csv", type=str, default="data/evaluation_corpus.csv", help="Ruta de salida del corpus")
    parser.add_argument("--output_dict", type=str, default="data/data_dictionary.md", help="Ruta del diccionario de datos")
    parser.add_argument("--total_samples", type=int, default=400, help="Numero total de definiciones a compilar")
    parser.add_argument("--dev_samples", type=int, default=100, help="Numero de definiciones para particion de desarrollo")
    parser.add_argument("--seed", type=int, default=42, help="Semilla determinista")
    args = parser.parse_args()

    logger.info("Compilando corpus de evaluacion desde %s con semilla %d", args.usf_csv, args.seed)
    df_eval, stats = collect_evaluation_samples(
        usf_csv_path=args.usf_csv,
        target_sample_size=args.total_samples,
        dev_size=args.dev_samples,
        random_seed=args.seed,
    )

    df_eval.to_csv(args.output_csv, index=False, encoding="utf-8")
    logger.info("Corpus de evaluacion guardado en: %s", args.output_csv)

    update_data_dictionary(args.output_dict, stats)

    logger.info("Total de muestras generadas: %d", stats["total_definitions"])
    logger.info("Particion dev: %d", stats["dev_definitions"])
    logger.info("Particion test: %d", stats["test_definitions"])


if __name__ == "__main__":
    main()
