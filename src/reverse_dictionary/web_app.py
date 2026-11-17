import logging
import os

import gradio as gr
import pandas as pd

from reverse_dictionary.application.agent import ReverseDictionaryAgent
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
)
from reverse_dictionary.infrastructure.loaders import CsvAssociationNormsLoader
from reverse_dictionary.infrastructure.nlp import SpacyEnglishPreprocessor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("web_app")


def build_initialized_agent(
    norms_path: str = "data/usf_norms.csv",
    models_dir: str = "models",
    seed: int = 42,
) -> ReverseDictionaryAgent:
    logger.info("Inicializando agente para la interfaz web...")
    loader = CsvAssociationNormsLoader(norms_path)
    edges = loader.load_edges()
    target_vocab = loader.load_vocabulary()

    container = AssociationGraphContainer.from_edges(edges)
    preprocessor = SpacyEnglishPreprocessor()

    agent = ReverseDictionaryAgent(preprocessor=preprocessor, target_vocabulary=target_vocab)

    agent.register_strategy(ClosenessCentralityRetrievalStrategy(container, "strength"))
    agent.register_strategy(ClosenessCentralityRetrievalStrategy(container, "frequency"))
    agent.register_strategy(BetweennessCentralityRetrievalStrategy(container, "strength"))
    agent.register_strategy(BetweennessCentralityRetrievalStrategy(container, "frequency"))
    agent.register_strategy(DegreeCentralityRetrievalStrategy(container, "strength"))
    agent.register_strategy(DirectedPageRankRetrievalStrategy(container))

    trainer = Node2VecTrainer(models_dir=models_dir)
    configs = [
        Node2VecConfig(name="node2vec_default", dimensions=64, walk_length=15, num_walks=10, p=1.0, q=1.0, seed=seed),
        Node2VecConfig(name="node2vec_homophily_bfs", dimensions=64, walk_length=15, num_walks=10, p=1.0, q=2.0, seed=seed),
        Node2VecConfig(name="node2vec_structural_dfs", dimensions=64, walk_length=20, num_walks=10, p=1.0, q=0.5, seed=seed),
    ]

    for cfg in configs:
        wv = trainer.train_or_load(container.undirected_graph, cfg)
        agent.register_strategy(Node2VecRetrievalStrategy(cfg.name, wv, target_vocab))

    closeness_best = ClosenessCentralityRetrievalStrategy(container, "strength")
    default_n2v = agent._strategies["node2vec_default"]
    agent.register_strategy(ReciprocalRankFusionStrategy(closeness_best, default_n2v))

    pretrained_strat = PretrainedEmbeddingRetrievalStrategy.load_from_gensim(
        model_name="glove-wiki-gigaword-50", target_vocabulary=target_vocab
    )
    agent.register_strategy(pretrained_strat)

    logger.info("Agente inicializado con %d estrategias", len(agent.available_strategies))
    return agent


def create_gradio_app(agent: ReverseDictionaryAgent) -> gr.Blocks:
    strategy_choices = list(agent.available_strategies)
    default_strategy = "fusion_rrf_cercania_grafo_reducido_strength__and__node2vec_default"
    if default_strategy not in strategy_choices:
        default_strategy = strategy_choices[0]

    def predict(definition: str, strategy: str, top_k: int):
        if not definition or not definition.strip():
            return pd.DataFrame(), "Por favor ingrese una definicion.", "0.0 ms"

        result = agent.query(definition, strategy, top_k=int(top_k))
        lemmas_text = ", ".join(result.query_lemmas) if result.query_lemmas else "Ningun lema reconocido en el grafo"
        exec_time_text = f"{result.execution_time_seconds * 1000.0:.2f} ms"

        data = [
            {"Rango": item.rank, "Palabra Candidata": item.word, "Puntaje": item.score}
            for item in result.candidates
        ]
        df = pd.DataFrame(data)
        return df, lemmas_text, exec_time_text

    examples = [
        [
            "the opening through which food is taken in and vocalizations emerge",
            default_strategy,
            5,
        ],
        [
            "a large natural stream of water flowing in a definite course",
            default_strategy,
            5,
        ],
        [
            "a nocturnal burrowing mammal of the grasslands of Africa that feeds on termites",
            default_strategy,
            5,
        ],
        [
            "a mechanical device that moves fluid by suction or pressure",
            "cercania_grafo_reducido_strength",
            5,
        ],
        [
            "the social event at which the ceremony of marriage is performed",
            "pagerank_personalizado_dirigido",
            5,
        ],
    ]

    with gr.Blocks(title="Diccionario Inverso (USF Free Association Norms)") as demo:
        gr.Markdown(
            """
            # Diccionario Inverso
            **Autor:** Osmar Ciau
            """
        )

        with gr.Row():
            with gr.Column(scale=2):
                input_def = gr.Textbox(
                    label="Definicion o descripcion en lenguaje natural (Ingles)",
                    placeholder="Ejemplo: the opening through which food is taken in and vocalizations emerge",
                    lines=3,
                )
                with gr.Row():
                    strategy_dropdown = gr.Dropdown(
                        label="Estrategia de Busqueda / Modelo",
                        choices=strategy_choices,
                        value=default_strategy,
                    )
                    top_k_slider = gr.Slider(
                        minimum=1, maximum=20, value=5, step=1, label="Top K Candidatos"
                    )
                search_btn = gr.Button("Buscar Palabra Candidata", variant="primary")

            with gr.Column(scale=2):
                results_table = gr.DataFrame(
                    label="Palabras Candidatas Recuperadas",
                    headers=["Rango", "Palabra Candidata", "Puntaje"],
                )
                with gr.Row():
                    lemmas_output = gr.Textbox(label="Lemas Reconocidos en el Grafo", interactive=False)
                    time_output = gr.Textbox(label="Tiempo de Inferencia", interactive=False)

        gr.Examples(
            examples=examples,
            inputs=[input_def, strategy_dropdown, top_k_slider],
            outputs=[results_table, lemmas_output, time_output],
            fn=predict,
            cache_examples=False,
        )

        search_btn.click(
            fn=predict,
            inputs=[input_def, strategy_dropdown, top_k_slider],
            outputs=[results_table, lemmas_output, time_output],
        )

    return demo


def main() -> None:
    agent = build_initialized_agent()
    demo = create_gradio_app(agent)
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False)


if __name__ == "__main__":
    main()

