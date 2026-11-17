# Agent Instructions: Reverse Dictionary System

## 1. Project Context & Objectives
You are an expert Python software engineer and NLP researcher working on a Reverse Dictionary system based on word association norms and graph theory.
The system takes natural language definitions or descriptions, retrieves concept lemmas, and ranks target candidate words using graph centrality metrics and graph embeddings (`node2vec`).
The primary dataset consists of the University of South Florida (USF) Free Association Norms (5,019 cue nodes, ~72,000 edges) evaluated on natural language definitions extracted from WordNet and CODWOE.

## 2. Global Constraints & Enforcement
- Under no circumstance should you write inline or block code comments (`#`) in any Python file or generated code. The code must be self-documenting through clear naming conventions and type annotations.
- Do not use emojis in commit messages, documentation, terminal outputs, logs, or user interfaces.
- Avoid stdout console printing (`print`). Use the standard `logging` library configured with structured formatters and appropriate log levels (`DEBUG`, `INFO`, `WARNING`, `ERROR`).
- Always set deterministic random seeds (`seed=42`) across graph random walks, train/test splits, and vector models to ensure strict reproducibility.
- Environment and dependency management must strictly use `uv`.

## 3. Onion Architecture Principles
All code must adhere to Onion Architecture (Ports and Adapters). Dependencies must flow strictly inward toward the domain layer.

### Directory Layout
```text
src/reverse_dictionary/
├── domain/
│   ├── models.py
│   └── protocols.py
├── application/
│   ├── agent.py
│   └── use_cases.py
├── infrastructure/
│   ├── loaders.py
│   ├── nlp.py
│   ├── graph_adapters.py
│   └── embedding_adapters.py
└── main.py
```

### Layer Responsibilities
- `domain/`: Contains enterprise business rules, immutable data transfer objects (`@dataclass(frozen=True)`), domain exceptions, and interface specifications using `typing.Protocol`. It must have zero third-party dependencies (no networkx, spacy, nltk, pandas, or numpy).
- `application/`: Coordinates domain models to execute user workflows, such as query evaluation, candidate ranking, and metric benchmarking ($p@1, p@3, p@5$, MRR, coverage). It depends only on domain protocols and models.
- `infrastructure/`: Implements adapters for external frameworks and libraries (networkx graph algorithms, spacy text preprocessing, gensim/node2vec vector generation, and CSV dataset loaders).
- `main.py` (Presentation / Entrypoint): Wires dependencies via inversion of control, registers available retrieval strategies into the `ReverseDictionaryAgent`, and runs benchmarks or interactive lookups.

## 4. Domain & Retrieval Rules
- Exclude input definition lemmas from candidate answer sets.
- Weights in shortest path algorithms represent distance/cost. Transform affinity scores accordingly: $1.0 / \text{frequency}$ or $1.0 - \text{association\_strength}$.
- To handle computational complexity during centrality calculations, always compute metrics over the 1-hop neighborhood subgraph centered around query lemmas rather than recomputing all-pairs paths over the full graph.
- Evaluation metrics must compute precision at $k$ ($p@1, p@3, p@5$), Mean Reciprocal Rank (MRR), and lexical coverage over segregated development and test sets.

## 5. Coding Standards & Best Practices
- Python 3.12+ type hints are mandatory across all function signatures, parameters, and return types (`typing`, `collections.abc`).
- Prefer pure functions and stateless transformation classes.
- Use dependency injection to pass adapters to application use cases.
- Handle IO and missing lexical items gracefully without unhandled runtime crashes.
