# Tabla Comparativa

| Modelo / Estrategia | Particion | p@1 | p@3 | p@5 | MRR | Cobertura | Tiempo Medio (s) |
|---|---|---|---|---|---|---|---|
| `cercania_grafo_reducido_strength` | test | 0.017 | 0.063 | 0.090 | 0.054 | 1.000 | 0.0417 |
| `cercania_grafo_reducido_frequency` | test | 0.020 | 0.063 | 0.110 | 0.056 | 1.000 | 0.0488 |
| `intermediacion_grafo_reducido_strength` | test | 0.027 | 0.057 | 0.090 | 0.053 | 0.987 | 0.0046 |
| `intermediacion_grafo_reducido_frequency` | test | 0.020 | 0.067 | 0.083 | 0.051 | 0.997 | 0.0045 |
| `grado_grafo_reducido_strength` | test | 0.027 | 0.070 | 0.137 | 0.071 | 1.000 | 0.0026 |
| `grado_grafo_reducido_frequency` | test | 0.023 | 0.077 | 0.140 | 0.071 | 1.000 | 0.0025 |
| `pagerank_personalizado_dirigido` | test | 0.007 | 0.033 | 0.043 | 0.024 | 0.993 | 0.0028 |
| `node2vec_default` | test | 0.007 | 0.020 | 0.027 | 0.019 | 1.000 | 0.0069 |
| `node2vec_homophily_bfs` | test | 0.003 | 0.027 | 0.040 | 0.020 | 1.000 | 0.0084 |
| `node2vec_structural_dfs` | test | 0.003 | 0.027 | 0.040 | 0.019 | 1.000 | 0.0089 |
| `fusion_rrf_cercania_grafo_reducido_strength__and__node2vec_default` | test | 0.060 | 0.100 | 0.120 | 0.088 | 1.000 | 0.0806 |
| `baseline_glove-wiki-gigaword-50` | test | 0.010 | 0.020 | 0.033 | 0.019 | 1.000 | 0.0065 |
| `node2vec_default` | dev | 0.000 | 0.020 | 0.040 | 0.012 | 1.000 | 0.0073 |
| `node2vec_homophily_bfs` | dev | 0.000 | 0.010 | 0.030 | 0.012 | 1.000 | 0.0066 |
| `node2vec_structural_dfs` | dev | 0.000 | 0.020 | 0.030 | 0.013 | 1.000 | 0.0064 |

---