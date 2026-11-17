import json
import logging
import os
from typing import Literal

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from reverse_dictionary.domain.models import (
    CandidateScore,
    GraphSummary,
    WeightScheme,
    WordAssociationEdge,
)
from reverse_dictionary.domain.protocols import (
    GraphAnalyzerProtocol,
    RetrievalStrategyProtocol,
)

logger = logging.getLogger("graph_adapters")


class AssociationGraphContainer:
    def __init__(self, undirected_graph: nx.Graph, directed_graph: nx.DiGraph) -> None:
        self._undirected_graph = undirected_graph
        self._directed_graph = directed_graph

    @property
    def undirected_graph(self) -> nx.Graph:
        return self._undirected_graph

    @property
    def directed_graph(self) -> nx.DiGraph:
        return self._directed_graph

    @classmethod
    def from_edges(
        cls, edges: tuple[WordAssociationEdge, ...]
    ) -> "AssociationGraphContainer":
        undirected_g = nx.Graph()
        directed_g = nx.DiGraph()

        for edge in edges:
            directed_g.add_edge(
                edge.source,
                edge.target,
                frequency=edge.frequency,
                forward_strength=edge.forward_strength,
                backward_strength=edge.backward_strength,
                dist_frequency=edge.distance_frequency,
                dist_strength=edge.distance_strength,
                affinity_frequency=float(edge.frequency),
                affinity_strength=edge.forward_strength,
            )

            if undirected_g.has_edge(edge.source, edge.target):
                existing_data = undirected_g[edge.source][edge.target]
                min_dist_freq = min(existing_data["dist_frequency"], edge.distance_frequency)
                min_dist_strength = min(existing_data["dist_strength"], edge.distance_strength)
                max_aff_freq = max(existing_data["affinity_frequency"], float(edge.frequency))
                max_aff_strength = max(existing_data["affinity_strength"], edge.forward_strength)

                undirected_g[edge.source][edge.target]["dist_frequency"] = min_dist_freq
                undirected_g[edge.source][edge.target]["dist_strength"] = min_dist_strength
                undirected_g[edge.source][edge.target]["affinity_frequency"] = max_aff_freq
                undirected_g[edge.source][edge.target]["affinity_strength"] = max_aff_strength
            else:
                undirected_g.add_edge(
                    edge.source,
                    edge.target,
                    dist_frequency=edge.distance_frequency,
                    dist_strength=edge.distance_strength,
                    affinity_frequency=float(edge.frequency),
                    affinity_strength=edge.forward_strength,
                )

        logger.info(
            "Grafo no dirigido construido: %d nodos, %d aristas",
            undirected_g.number_of_nodes(),
            undirected_g.number_of_edges(),
        )
        logger.info(
            "Grafo dirigido construido: %d nodos, %d aristas",
            directed_g.number_of_nodes(),
            directed_g.number_of_edges(),
        )

        return cls(undirected_graph=undirected_g, directed_graph=directed_g)


class NetworkxGraphAnalyzer(GraphAnalyzerProtocol):
    def __init__(self, graph_container: AssociationGraphContainer) -> None:
        self._container = graph_container

    def get_summary(self) -> GraphSummary:
        graph = self._container.undirected_graph
        node_count = graph.number_of_nodes()
        edge_count = graph.number_of_edges()

        if node_count == 0:
            return GraphSummary(
                node_count=0,
                edge_count=0,
                average_degree=0.0,
                connected_components_count=0,
                largest_component_size=0,
                average_clustering_coefficient=0.0,
            )

        degrees = [deg for _, deg in graph.degree()]
        average_degree = round(float(np.mean(degrees)), 4)

        connected_components = list(nx.connected_components(graph))
        components_count = len(connected_components)
        largest_component_size = len(max(connected_components, key=len)) if connected_components else 0

        avg_clustering = round(float(nx.average_clustering(graph)), 4)

        summary = GraphSummary(
            node_count=node_count,
            edge_count=edge_count,
            average_degree=average_degree,
            connected_components_count=components_count,
            largest_component_size=largest_component_size,
            average_clustering_coefficient=avg_clustering,
        )

        logger.info(
            "Estadisticas calculadas: %d nodos, %d aristas, grado medio %.2f, %d componentes (mayor: %d), clustering %.4f",
            summary.node_count,
            summary.edge_count,
            summary.average_degree,
            summary.connected_components_count,
            summary.largest_component_size,
            summary.average_clustering_coefficient,
        )

        return summary

    def export_degree_distribution_plot(self, output_image_path: str) -> None:
        os.makedirs(os.path.dirname(output_image_path), exist_ok=True)
        graph = self._container.undirected_graph

        degrees = [deg for _, deg in graph.degree() if deg > 0]
        if not degrees:
            logger.warning("No hay grados disponibles para graficar")
            return

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        ax1.hist(degrees, bins=60, color="steelblue", edgecolor="black", alpha=0.7)
        ax1.set_title("Distribucion de Grado (Escala Lineal)")
        ax1.set_xlabel("Grado (k)")
        ax1.set_ylabel("Frecuencia (conteo de nodos)")
        ax1.grid(True, linestyle="--", alpha=0.5)

        degree_counts: dict[int, int] = {}
        for d in degrees:
            degree_counts[d] = degree_counts.get(d, 0) + 1

        sorted_degrees = sorted(degree_counts.keys())
        total_nodes = len(degrees)
        probabilities = [degree_counts[k] / total_nodes for k in sorted_degrees]

        ax2.loglog(sorted_degrees, probabilities, "bo", markersize=4, alpha=0.6)
        ax2.set_title("Distribucion de Grado (Escala Log-Log: Ley de Potencia)")
        ax2.set_xlabel("Grado (log k)")
        ax2.set_ylabel("P(k) (log probabilidad)")
        ax2.grid(True, which="both", linestyle="--", alpha=0.5)

        plt.tight_layout()
        plt.savefig(output_image_path, dpi=300)
        plt.close(fig)

        logger.info("Grafica de distribucion de grado exportada en: %s", output_image_path)

    def save_summary_json(self, output_json_path: str) -> None:
        os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
        summary = self.get_summary()
        data = {
            "node_count": summary.node_count,
            "edge_count": summary.edge_count,
            "average_degree": summary.average_degree,
            "connected_components_count": summary.connected_components_count,
            "largest_component_size": summary.largest_component_size,
            "average_clustering_coefficient": summary.average_clustering_coefficient,
        }
        with open(output_json_path, "w", encoding="utf-8") as fp:
            json.dump(data, fp, indent=4)
        logger.info("Resumen estadistico guardado en: %s", output_json_path)


def extract_1hop_neighborhood_subgraph(
    graph: nx.Graph | nx.DiGraph, query_lemmas: tuple[str, ...]
) -> tuple[nx.Graph | nx.DiGraph, set[str]]:
    valid_query_nodes = set(node for node in query_lemmas if graph.has_node(node))
    if not valid_query_nodes:
        return graph.__class__(), set()

    neighborhood_nodes: set[str] = set(valid_query_nodes)
    for node in valid_query_nodes:
        if isinstance(graph, nx.DiGraph):
            neighborhood_nodes.update(graph.successors(node))
            neighborhood_nodes.update(graph.predecessors(node))
        else:
            neighborhood_nodes.update(graph.neighbors(node))

    subgraph = graph.subgraph(neighborhood_nodes).copy()
    candidate_nodes = neighborhood_nodes.difference(valid_query_nodes)
    return subgraph, candidate_nodes


class ClosenessCentralityRetrievalStrategy(RetrievalStrategyProtocol):
    def __init__(
        self,
        container: AssociationGraphContainer,
        weight_scheme: WeightScheme = "strength",
    ) -> None:
        self._container = container
        self._weight_scheme = weight_scheme
        self._weight_attr = "dist_strength" if weight_scheme == "strength" else "dist_frequency"

    @property
    def strategy_name(self) -> str:
        return f"cercania_grafo_reducido_{self._weight_scheme}"

    def retrieve_candidates(
        self, query_lemmas: tuple[str, ...], top_k: int = 10
    ) -> tuple[CandidateScore, ...]:
        subgraph, candidate_nodes = extract_1hop_neighborhood_subgraph(
            self._container.undirected_graph, query_lemmas
        )
        if not candidate_nodes:
            return ()

        closeness_map = nx.closeness_centrality(subgraph, distance=self._weight_attr)

        filtered_scores = [
            (node, closeness_map[node])
            for node in candidate_nodes
            if node in closeness_map and closeness_map[node] > 0.0
        ]
        filtered_scores.sort(key=lambda item: item[1], reverse=True)

        candidates = [
            CandidateScore(word=node, score=round(float(score), 6), rank=idx + 1)
            for idx, (node, score) in enumerate(filtered_scores[:top_k])
        ]
        return tuple(candidates)


class BetweennessCentralityRetrievalStrategy(RetrievalStrategyProtocol):
    def __init__(
        self,
        container: AssociationGraphContainer,
        weight_scheme: WeightScheme = "strength",
    ) -> None:
        self._container = container
        self._weight_scheme = weight_scheme
        self._weight_attr = "dist_strength" if weight_scheme == "strength" else "dist_frequency"

    @property
    def strategy_name(self) -> str:
        return f"intermediacion_grafo_reducido_{self._weight_scheme}"

    def retrieve_candidates(
        self, query_lemmas: tuple[str, ...], top_k: int = 10
    ) -> tuple[CandidateScore, ...]:
        subgraph, candidate_nodes = extract_1hop_neighborhood_subgraph(
            self._container.undirected_graph, query_lemmas
        )
        if not candidate_nodes:
            return ()

        valid_sources = [node for node in query_lemmas if subgraph.has_node(node)]
        if len(valid_sources) >= 2 and len(subgraph) < 1500:
            betweenness_map = nx.betweenness_centrality_subset(
                subgraph,
                sources=valid_sources,
                targets=list(candidate_nodes),
                weight=self._weight_attr,
            )
        else:
            betweenness_map = nx.betweenness_centrality(subgraph, weight=self._weight_attr)

        filtered_scores = [
            (node, betweenness_map[node])
            for node in candidate_nodes
            if node in betweenness_map and betweenness_map[node] > 0.0
        ]
        filtered_scores.sort(key=lambda item: item[1], reverse=True)

        candidates = [
            CandidateScore(word=node, score=round(float(score), 6), rank=idx + 1)
            for idx, (node, score) in enumerate(filtered_scores[:top_k])
        ]
        return tuple(candidates)


class DegreeCentralityRetrievalStrategy(RetrievalStrategyProtocol):
    def __init__(
        self,
        container: AssociationGraphContainer,
        weight_scheme: WeightScheme = "strength",
    ) -> None:
        self._container = container
        self._weight_scheme = weight_scheme
        self._affinity_attr = "affinity_strength" if weight_scheme == "strength" else "affinity_frequency"

    @property
    def strategy_name(self) -> str:
        return f"grado_grafo_reducido_{self._weight_scheme}"

    def retrieve_candidates(
        self, query_lemmas: tuple[str, ...], top_k: int = 10
    ) -> tuple[CandidateScore, ...]:
        subgraph, candidate_nodes = extract_1hop_neighborhood_subgraph(
            self._container.undirected_graph, query_lemmas
        )
        if not candidate_nodes:
            return ()

        valid_query_nodes = set(node for node in query_lemmas if subgraph.has_node(node))
        scores: dict[str, float] = {}

        for candidate in candidate_nodes:
            accumulated_affinity = 0.0
            for query_node in valid_query_nodes:
                if subgraph.has_edge(candidate, query_node):
                    edge_data = subgraph[candidate][query_node]
                    accumulated_affinity += float(edge_data.get(self._affinity_attr, 1.0))
            if accumulated_affinity > 0.0:
                scores[candidate] = accumulated_affinity

        sorted_candidates = sorted(scores.items(), key=lambda item: item[1], reverse=True)

        candidates = [
            CandidateScore(word=node, score=round(float(score), 6), rank=idx + 1)
            for idx, (node, score) in enumerate(sorted_candidates[:top_k])
        ]
        return tuple(candidates)


class DirectedPageRankRetrievalStrategy(RetrievalStrategyProtocol):
    def __init__(self, container: AssociationGraphContainer, alpha: float = 0.85) -> None:
        self._container = container
        self._alpha = alpha

    @property
    def strategy_name(self) -> str:
        return "pagerank_personalizado_dirigido"

    def retrieve_candidates(
        self, query_lemmas: tuple[str, ...], top_k: int = 10
    ) -> tuple[CandidateScore, ...]:
        subgraph, candidate_nodes = extract_1hop_neighborhood_subgraph(
            self._container.directed_graph, query_lemmas
        )
        if not candidate_nodes or len(subgraph) == 0:
            return ()

        valid_query_nodes = [node for node in query_lemmas if subgraph.has_node(node)]
        if not valid_query_nodes:
            return ()

        uniform_weight = 1.0 / len(valid_query_nodes)
        personalization = {node: (uniform_weight if node in valid_query_nodes else 0.0) for node in subgraph.nodes()}

        try:
            pagerank_map = nx.pagerank(
                subgraph,
                alpha=self._alpha,
                personalization=personalization,
                weight="affinity_strength",
                max_iter=100,
            )
        except nx.PowerIterationFailedConvergence:
            pagerank_map = nx.pagerank(subgraph, alpha=self._alpha, max_iter=200)

        filtered_scores = [
            (node, pagerank_map[node])
            for node in candidate_nodes
            if node in pagerank_map and pagerank_map[node] > 0.0
        ]
        filtered_scores.sort(key=lambda item: item[1], reverse=True)

        candidates = [
            CandidateScore(word=node, score=round(float(score), 6), rank=idx + 1)
            for idx, (node, score) in enumerate(filtered_scores[:top_k])
        ]
        return tuple(candidates)
