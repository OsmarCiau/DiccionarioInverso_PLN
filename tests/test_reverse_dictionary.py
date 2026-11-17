import dataclasses
import os
import pathlib
import unittest

from reverse_dictionary.application.agent import ReverseDictionaryAgent
from reverse_dictionary.domain.models import (
    CandidateScore,
    EmptyQueryError,
    EvaluationSample,
    ReverseDictionaryDomainError,
    WordAssociationEdge,
)
from reverse_dictionary.infrastructure.graph_adapters import (
    AssociationGraphContainer,
    ClosenessCentralityRetrievalStrategy,
    extract_1hop_neighborhood_subgraph,
)
from reverse_dictionary.infrastructure.nlp import SpacyEnglishPreprocessor


class TestDomainLayer(unittest.TestCase):
    def test_domain_models_immutability(self) -> None:
        candidate = CandidateScore(word="elephant", score=0.95, rank=1)
        with self.assertRaises(dataclasses.FrozenInstanceError):
            candidate.score = 0.50

    def test_domain_exceptions_hierarchy(self) -> None:
        self.assertTrue(issubclass(EmptyQueryError, ReverseDictionaryDomainError))


class TestInfrastructureLayer(unittest.TestCase):
    def test_spacy_preprocessor_lemmas(self) -> None:
        preprocessor = SpacyEnglishPreprocessor()
        text = "The black cats are running very quickly in the garden."
        lemmas = preprocessor.extract_lemmas(text)
        self.assertIn("black", lemmas)
        self.assertIn("cat", lemmas)
        self.assertIn("run", lemmas)
        self.assertIn("garden", lemmas)
        self.assertNotIn("the", lemmas)
        self.assertNotIn("are", lemmas)

    def test_spacy_valid_query_lemmas_filters_stopwords_with_punctuation(self) -> None:
        preprocessor = SpacyEnglishPreprocessor()
        vocab = {"biology", "the", "process", "of", "an", "organism", "grow"}
        text = "(biology) the process of an organism growing"
        lemmas = preprocessor.extract_valid_query_lemmas(text, vocab)
        self.assertIn("biology", lemmas)
        self.assertIn("process", lemmas)
        self.assertIn("organism", lemmas)
        self.assertIn("grow", lemmas)
        self.assertNotIn("the", lemmas)
        self.assertNotIn("of", lemmas)
        self.assertNotIn("an", lemmas)

    def test_graph_container_and_1hop(self) -> None:
        edges = (
            WordAssociationEdge("river", "water", 50, 0.5, 0.2, 0.02, 0.5),
            WordAssociationEdge("water", "drink", 40, 0.4, 0.1, 0.025, 0.6),
            WordAssociationEdge("drink", "cup", 20, 0.2, 0.05, 0.05, 0.8),
            WordAssociationEdge("sun", "hot", 30, 0.3, 0.1, 0.033, 0.7),
        )
        container = AssociationGraphContainer.from_edges(edges)
        subgraph, candidates = extract_1hop_neighborhood_subgraph(
            container.undirected_graph, ("river",)
        )
        self.assertIn("water", candidates)
        self.assertNotIn("river", candidates)
        self.assertNotIn("sun", candidates)


class TestApplicationLayer(unittest.TestCase):
    def test_agent_query_flow(self) -> None:
        edges = (
            WordAssociationEdge("trunk", "elephant", 50, 0.8, 0.3, 0.02, 0.2),
            WordAssociationEdge("africa", "elephant", 45, 0.7, 0.2, 0.022, 0.3),
            WordAssociationEdge("trunk", "tree", 30, 0.4, 0.1, 0.033, 0.6),
        )
        container = AssociationGraphContainer.from_edges(edges)
        preprocessor = SpacyEnglishPreprocessor()
        vocab = {"trunk", "elephant", "africa", "tree"}

        agent = ReverseDictionaryAgent(preprocessor=preprocessor, target_vocabulary=vocab)
        strategy = ClosenessCentralityRetrievalStrategy(container, "strength")
        agent.register_strategy(strategy)

        result = agent.query(
            "large animal in africa with a long trunk",
            strategy_name=strategy.strategy_name,
            top_k=3,
        )

        self.assertGreater(len(result.candidates), 0)
        self.assertEqual(result.candidates[0].word, "elephant")
        self.assertGreater(result.execution_time_seconds, 0.0)

    def test_empty_query_raises_error(self) -> None:
        preprocessor = SpacyEnglishPreprocessor()
        agent = ReverseDictionaryAgent(preprocessor=preprocessor, target_vocabulary=set())
        with self.assertRaises(EmptyQueryError):
            agent.query("   ", "cercania_grafo_reducido_strength")


class TestAgentsMdRulesEnforcement(unittest.TestCase):
    def test_no_inline_or_block_code_comments_in_python_files(self) -> None:
        root_dir = pathlib.Path(__file__).parent.parent
        src_dir = root_dir / "src"
        py_files = list(src_dir.rglob("*.py"))
        py_files.append(root_dir / "main.py")
        py_files.extend(list((root_dir / "data").glob("*.py")))

        violating_lines: list[tuple[str, int, str]] = []
        for file_path in py_files:
            text = file_path.read_text(encoding="utf-8")
            for idx, line in enumerate(text.splitlines(), 1):
                stripped = line.strip()
                if stripped.startswith("#"):
                    if not (stripped.startswith("# ") or stripped.startswith("## ") or stripped.startswith("### ")):
                        violating_lines.append((str(file_path), idx, line))

        self.assertEqual(
            len(violating_lines),
            0,
            f"Se encontraron comentarios de codigo (#) que violan AGENTS.md: {violating_lines}",
        )


if __name__ == "__main__":
    unittest.main()

