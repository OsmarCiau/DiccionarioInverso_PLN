import argparse
import logging
import os
import time

import pandas as pd

from reverse_dictionary.application.agent import ReverseDictionaryAgent
from reverse_dictionary.application.use_cases import (
    BenchmarkEvaluationUseCase,
    ErrorAnalysisUseCase,
)
from reverse_dictionary.domain.models import Node2VecConfig
from reverse_dictionary.infrastructure.embedding_adapters import (
    Node2VecRetrievalStrategy,
    Node2VecTrainer,
    PretrainedEmbeddingRetrievalStrategy,
    ReciprocalRankFusionStrategy,
)
from reverse_dictionary.infrastructure.graph_adapters import (
    AssociationGraphContainer,
    BetweennessCentralityRetrievalStrategy,
    ClosenessCentralityRetrievalStrategy,
    DegreeCentralityRetrievalStrategy,
    DirectedPageRankRetrievalStrategy,
    NetworkxGraphAnalyzer,
)
from reverse_dictionary.infrastructure.loaders import (
    CsvAssociationNormsLoader,
    CsvEvaluationCorpusLoader,
)
from reverse_dictionary.infrastructure.nlp import SpacyEnglishPreprocessor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("main")


def export_benchmark_markdown_table(
    metrics_list: list, output_md_path: str
) -> None:
    os.makedirs(os.path.dirname(output_md_path), exist_ok=True)

    header = """# Tabla Comparativa de Resultados: Diccionario Inverso (USF Norms)

Evaluacion realizada sobre el conjunto de prueba (`test`, 300 definiciones) y conjunto de desarrollo (`dev`, 100 definiciones) con semillas fijas (`seed=42`).

| Modelo / Estrategia | Particion | p@1 | p@3 | p@5 | MRR | Cobertura | Tiempo Medio (s) |
|---|---|---|---|---|---|---|---|
"""
    rows = []
    for m in metrics_list:
        row = (
            f"| `{m.strategy_name}` | {m.partition} | "
            f"{m.precision_at_1:.3f} | {m.precision_at_3:.3f} | {m.precision_at_5:.3f} | "
            f"{m.mean_reciprocal_rank:.3f} | {m.coverage_ratio:.3f} | {m.average_execution_time_seconds:.4f} |"
        )
        rows.append(row)

    reference_note = """

---

### Comparacion contra Resultados de Referencia de Clase (NAP - Tabla 1)

| Modelo de Referencia (NAP en Clase) | Ponderacion | p@1 | p@3 | p@5 |
|---|---|---|---|---|
| Intermediacion por subconjunto, grafo completo | Asociacion | 0.246 | 0.406 | 0.438 |
| Cercania, grafo reducido | Asociacion | 0.221 | 0.428 | 0.518 |
| Cercania, grafo reducido | Frecuencia | 0.196 | 0.355 | 0.514 |
| Grado, grafo reducido | Asociacion | 0.109 | 0.239 | 0.362 |
| Comunicabilidad, grafo reducido | Asociacion | 0.116 | 0.228 | 0.319 |
"""

    content = header + "\n".join(rows) + reference_note
    with open(output_md_path, "w", encoding="utf-8") as fp:
        fp.write(content)
    logger.info("Tabla comparativa exportada en: %s", output_md_path)


def export_error_analysis_markdown(
    error_cases: list, output_md_path: str
) -> None:
    os.makedirs(os.path.dirname(output_md_path), exist_ok=True)

    header = """# Analisis de Casos de Error y Discusion Cualitativa

Analisis de 15 casos de error clasificados por causa metodologica, conforme a la seccion 4.6 de las especificaciones del proyecto.

| No. | Concepto Objetivo | Definicion de Entrada | Lemas Extraidos | Top-5 Candidatos Predichos | Causa del Error |
|---|---|---|---|---|---|
"""
    rows = []
    for idx, case in enumerate(error_cases, 1):
        lemmas_str = ", ".join(case.extracted_lemmas) if case.extracted_lemmas else "ninguno"
        preds_str = ", ".join(case.predicted_top_5) if case.predicted_top_5 else "ninguno"
        row = (
            f"| {idx} | **{case.concept}** | *{case.definition}* | "
            f"`{lemmas_str}` | `{preds_str}` | {case.error_category} |"
        )
        rows.append(row)

    discussion = """

---

## Discusion y Respuestas a Preguntas Guia

### 1. ¿Que informacion se pierde al modelar las asociaciones, que son dirigidas, con un grafo no dirigido?
En las normas de asociacion libre, la asimetria cognitiva es fundamental: un estimulo $A$ evoca una respuesta $B$ con mucha mayor probabilidad que la inversa (por ejemplo, `zebra` $\\to$ `stripes` con alta fuerza, pero `stripes` suele evocar `lines`, `candy` o `tiger`, rara vez `zebra`). Al colapsar el grafo en no dirigido, una palabra muy general con miles de respuestas entrantes puede «inundar» artificialmente los caminos mas cortos hacia conceptos especificos. Al evaluar el **PageRank Personalizado sobre el grafo dirigido**, se preserva el flujo direccional genuino de la asociacion humana, evitando falsos atajos semanticos.

### 2. ¿Por que la cercania en el grafo reducido mejora p@5 respecto a la intermediacion, pero no necesariamente p@1?
La centralidad de cercania suma las distancias geodesicas hacia todos los lemas de la definicion. Esto favorece fuertemente a candidatos situados en el «centro de gravedad» de los conceptos de entrada, logrando que el concepto correcto casi siempre caiga dentro del vecindario top 5 ($p@5$). Por el contrario, la intermediacion (*betweenness*) exige que el nodo este exactamente en el camino critico entre pares de lemas: cuando un lema clave conecta de forma unica y directa con la palabra objetivo, la intermediacion puede asignarle el rango 1 ($p@1$), pero penaliza a conceptos que estan equidistantemente cercanos sin ser cuellos de botella topologicos.

### 3. ¿Que ponderacion funciona mejor y por que? ¿Como influye la transformacion de similitud a distancia?
La ponderacion de **Fuerza de Asociacion** ($1.0 - FSG$) supera consistentemente a la frecuencia simple ($1/f$). La fuerza de asociacion es una probabilidad condicional normalizada ($P(\\text{respuesta}|\\text{estimulo}) = \\#P / \\#G$) que desacopla el tamano de muestra del experimento, mientras que la frecuencia cruda es sensible a variaciones en el numero de participantes. La transformacion lineal $1.0 - FSG$ acota los costos de transicion entre $0$ y $1$, permitiendo que Dijkstra compute sumas de penalizacion proporcionadas a la distancia cognitiva.
"""

    content = header + "\n".join(rows) + discussion
    with open(output_md_path, "w", encoding="utf-8") as fp:
        fp.write(content)
    logger.info("Reporte de analisis de errores exportado en: %s", output_md_path)


def run_pipeline(
    norms_path: str = "data/usf_norms.csv",
    eval_path: str = "data/evaluation_corpus.csv",
    reports_dir: str = "reports",
    models_dir: str = "models",
    random_seed: int = 42,
) -> None:
    norms_loader = CsvAssociationNormsLoader(norms_path)
    edges = norms_loader.load_edges()
    target_vocab = norms_loader.load_vocabulary()

    container = AssociationGraphContainer.from_edges(edges)

    analyzer = NetworkxGraphAnalyzer(container)
    figures_dir = os.path.join(reports_dir, "figures")
    plot_path = os.path.join(figures_dir, "degree_distribution.png")
    json_path = os.path.join(reports_dir, "graph_characterization.json")

    analyzer.export_degree_distribution_plot(plot_path)
    analyzer.save_summary_json(json_path)

    preprocessor = SpacyEnglishPreprocessor()
    agent = ReverseDictionaryAgent(preprocessor=preprocessor, target_vocabulary=target_vocab)

    agent.register_strategy(ClosenessCentralityRetrievalStrategy(container, "strength"))
    agent.register_strategy(ClosenessCentralityRetrievalStrategy(container, "frequency"))
    agent.register_strategy(BetweennessCentralityRetrievalStrategy(container, "strength"))
    agent.register_strategy(BetweennessCentralityRetrievalStrategy(container, "frequency"))
    agent.register_strategy(DegreeCentralityRetrievalStrategy(container, "strength"))
    agent.register_strategy(DegreeCentralityRetrievalStrategy(container, "frequency"))
    agent.register_strategy(DirectedPageRankRetrievalStrategy(container))

    configs = [
        Node2VecConfig(name="node2vec_default", dimensions=64, walk_length=15, num_walks=10, p=1.0, q=1.0, seed=random_seed),
        Node2VecConfig(name="node2vec_homophily_bfs", dimensions=64, walk_length=15, num_walks=10, p=1.0, q=2.0, seed=random_seed),
        Node2VecConfig(name="node2vec_structural_dfs", dimensions=64, walk_length=20, num_walks=10, p=1.0, q=0.5, seed=random_seed),
    ]

    trainer = Node2VecTrainer(models_dir=models_dir)
    for cfg in configs:
        wv = trainer.train_or_load(container.undirected_graph, cfg)
        strat = Node2VecRetrievalStrategy(
            strategy_name=cfg.name,
            keyed_vectors=wv,
            target_vocabulary=target_vocab,
        )
        agent.register_strategy(strat)

    closeness_best = ClosenessCentralityRetrievalStrategy(container, "strength")
    default_n2v = agent._strategies["node2vec_default"]
    fusion_strat = ReciprocalRankFusionStrategy(strategy_a=closeness_best, strategy_b=default_n2v)
    agent.register_strategy(fusion_strat)

    pretrained_strat = PretrainedEmbeddingRetrievalStrategy.load_from_gensim(
        model_name="glove-wiki-gigaword-50", target_vocabulary=target_vocab
    )
    agent.register_strategy(pretrained_strat)

    eval_loader = CsvEvaluationCorpusLoader(eval_path)
    test_samples = eval_loader.load_samples(partition="test")
    dev_samples = eval_loader.load_samples(partition="dev")

    benchmark_use_case = BenchmarkEvaluationUseCase(agent)
    all_metrics = []

    logger.info("Evaluando en conjunto de prueba (TEST: %d muestras)...", len(test_samples))
    for strat_name in agent.available_strategies:
        m_test = benchmark_use_case.evaluate_strategy(strat_name, test_samples, partition_label="test")
        all_metrics.append(m_test)

    logger.info("Evaluando en conjunto de desarrollo (DEV: %d muestras)...", len(dev_samples))
    for strat_name in ["node2vec_default", "node2vec_homophily_bfs", "node2vec_structural_dfs"]:
        m_dev = benchmark_use_case.evaluate_strategy(strat_name, dev_samples, partition_label="dev")
        all_metrics.append(m_dev)

    metrics_df = pd.DataFrame([
        {
            "strategy": m.strategy_name,
            "partition": m.partition,
            "p@1": m.precision_at_1,
            "p@3": m.precision_at_3,
            "p@5": m.precision_at_5,
            "mrr": m.mean_reciprocal_rank,
            "coverage": m.coverage_ratio,
            "avg_time_sec": m.average_execution_time_seconds,
        }
        for m in all_metrics
    ])
    csv_report_path = os.path.join(reports_dir, "benchmark_results.csv")
    metrics_df.to_csv(csv_report_path, index=False)
    logger.info("Metricas guardadas en: %s", csv_report_path)

    md_table_path = os.path.join(reports_dir, "benchmark_table.md")
    export_benchmark_markdown_table(all_metrics, md_table_path)

    error_use_case = ErrorAnalysisUseCase(agent)
    best_strategy = "cercania_grafo_reducido_strength"
    error_cases = list(error_use_case.extract_error_cases(best_strategy, test_samples, max_cases=15))
    error_md_path = os.path.join(reports_dir, "error_analysis_15_cases.md")
    export_error_analysis_markdown(error_cases, error_md_path)


def interactive_lookup(
    agent: ReverseDictionaryAgent,
) -> None:
    logger.info("Modo interactivo iniciado. Ingrese una definicion en ingles:")
    sample_queries = [
        "the opening through which food is taken in and vocalizations emerge",
        "a large body of water surrounded by land",
        "a portable device used to make phone calls and send messages",
    ]
    for q in sample_queries:
        logger.info("Consulta: '%s'", q)
        for strat in ["cercania_grafo_reducido_strength", "node2vec_default", "pagerank_personalizado_dirigido"]:
            res = agent.query(q, strat, top_k=5)
            cands = [f"{c.word} ({c.score})" for c in res.candidates]
            logger.info("  Estrategia [%s]: %s (tiempo: %.4fs)", strat, ", ".join(cands), res.execution_time_seconds)


def main() -> None:
    parser = argparse.ArgumentParser(description="Diccionario Inverso basado en USF y Grafos.")
    parser.add_argument("--mode", type=str, default="benchmark", choices=["benchmark", "interactive", "web", "all"])
    parser.add_argument("--norms_csv", type=str, default="data/usf_norms.csv")
    parser.add_argument("--eval_csv", type=str, default="data/evaluation_corpus.csv")
    parser.add_argument("--reports_dir", type=str, default="reports")
    parser.add_argument("--models_dir", type=str, default="models")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.mode in ["benchmark", "all"]:
        run_pipeline(
            norms_path=args.norms_csv,
            eval_path=args.eval_csv,
            reports_dir=args.reports_dir,
            models_dir=args.models_dir,
            random_seed=args.seed,
        )

    if args.mode in ["interactive", "all"]:
        loader = CsvAssociationNormsLoader(args.norms_csv)
        container = AssociationGraphContainer.from_edges(loader.load_edges())
        agent = ReverseDictionaryAgent(
            preprocessor=SpacyEnglishPreprocessor(),
            target_vocabulary=loader.load_vocabulary(),
        )
        agent.register_strategy(ClosenessCentralityRetrievalStrategy(container, "strength"))
        agent.register_strategy(DirectedPageRankRetrievalStrategy(container))
        trainer = Node2VecTrainer(models_dir=args.models_dir)
        cfg = Node2VecConfig(name="node2vec_default", dimensions=64, walk_length=15, num_walks=10, p=1.0, q=1.0, seed=args.seed)
        wv = trainer.train_or_load(container.undirected_graph, cfg)
        agent.register_strategy(Node2VecRetrievalStrategy("node2vec_default", wv, loader.load_vocabulary()))
        interactive_lookup(agent)

    if args.mode == "web":
        from reverse_dictionary.web_app import build_initialized_agent, create_gradio_app

        agent = build_initialized_agent(
            norms_path=args.norms_csv,
            models_dir=args.models_dir,
            seed=args.seed,
        )
        demo = create_gradio_app(agent)
        demo.launch(server_name="127.0.0.1", server_port=7860, share=False)


if __name__ == "__main__":
    main()

