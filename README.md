# Reverse Dictionary System (USF Word Association Norms & Graph Theory)

- **Institucion:** Universidad Autonoma de Yucatan (UADY) - Facultad de Matematicas
- **Curso:** Procesamiento de Lenguaje Natural (PLN)
- **Autor:** Osmar Ruben Ciau Martin
- **Profesor:** Dr. Jorge Carlos Reyes Magaña
- **Fecha de Entrega:** 10 de noviembre de 2026
- **Modalidad:** Modalidad B (Recurso léxico en inglés: USF Free Association Norms)

---

## 2. Recursos Léxicos y Conjuntos de Datos

### 2.1 Normas de Asociación Libre USF (`data/usf_norms.csv`)
* **Fuente:** University of South Florida Free Association Norms (Nelson, McEvoy & Schreiber, 2004).
* **Pares asociativos:** 72,151 pares válidos.
* **Nodos en el grafo:** 10,599 palabras (5,017 estímulos únicos y 10,450 respuestas únicas).
* **Ponderaciones implementadas:**
  * **Distancia por frecuencia:** $d_{frec} = 1.0 / \text{frecuencia}$
  * **Distancia por fuerza de asociación:** $d_{asoc} = 1.0 - FSG$

### 2.2 Corpus de Evaluación (`data/evaluation_corpus.csv`)
* **Fuente:** WordNet (`nltk.corpus.wordnet`).
* **Total de definiciones:** 400 definiciones en lenguaje natural en inglés.
* **Partición:** 100 muestras en `dev` (desarrollo) y 300 muestras en `test` (prueba final).
* **Control de fugas léxicas:** 0% de fuga. Se auditó con `PorterStemmer` para garantizar que ni el concepto ni su raíz morfológica aparezcan dentro de la definición.
* **Cobertura en el grafo:** 100.0% de las definiciones cuentan con lemas en el vocabulario de USF (promedio: 4.70 lemas útiles por definición).

---

## 3. Estadísticas Estructurales del Grafo USF

Generadas en `reports/graph_characterization.json` y graficadas en `reports/figures/degree_distribution.png`:

| Metrica Estructural | Valor |
|---|---|
| Nodos ($|V|$) | 10,599 |
| Aristas no dirigidas ($|E|$) | 63,768 |
| Aristas dirigidas ($|E_{dir}|$) | 72,151 |
| Grado promedio ($\langle k \rangle$) | 12.0328 |
| Componentes conexas | 1 (100% conexo, componente gigante de 10,599 nodos) |
| Coeficiente de agrupamiento ($\langle C \rangle$) | 0.1191 (topología de mundo pequeño) |

---

## 4. Resultados del Benchmark y Tabla Comparativa

Evaluación realizada sobre el conjunto de prueba (`test`, 300 definiciones) con semilla fija `seed=42`:

| Modelo / Estrategia | Partición | p@1 | p@3 | p@5 | MRR | Cobertura | Tiempo Medio (s) |
|---|---|---|---|---|---|---|---|
| `grado_grafo_reducido_frequency` | test | 0.023 | 0.077 | **0.140** | 0.071 | 1.000 | 0.0027 |
| `grado_grafo_reducido_strength` | test | 0.027 | 0.070 | **0.137** | 0.071 | 1.000 | 0.0025 |
| `fusion_rrf_cercania_strength__and__node2vec` | test | **0.060** | **0.100** | **0.120** | **0.088** | 1.000 | 0.0806 |
| `baseline_glove-wiki-gigaword-50` | test | 0.010 | 0.020 | 0.033 | 0.019 | 1.000 | 0.0065 |
| `cercania_grafo_reducido_frequency` | test | 0.020 | 0.063 | 0.110 | 0.056 | 1.000 | 0.0488 |
| `intermediacion_grafo_reducido_strength` | test | 0.027 | 0.057 | 0.097 | 0.053 | 0.987 | 0.0042 |
| `cercania_grafo_reducido_strength` | test | 0.017 | 0.063 | 0.090 | 0.054 | 1.000 | 0.0423 |
| `intermediacion_grafo_reducido_frequency` | test | 0.020 | 0.067 | 0.083 | 0.051 | 0.997 | 0.0046 |
| `pagerank_personalizado_dirigido` | test | 0.007 | 0.033 | 0.043 | 0.024 | 0.993 | 0.0028 |
| `node2vec_homophily_bfs` ($q=2.0$) | test | 0.003 | 0.027 | 0.040 | 0.020 | 1.000 | 0.0061 |
| `node2vec_structural_dfs` ($q=0.5$) | test | 0.003 | 0.027 | 0.040 | 0.019 | 1.000 | 0.0078 |
| `node2vec_default` ($p=1.0, q=1.0$) | test | 0.007 | 0.020 | 0.027 | 0.019 | 1.000 | 0.0055 |

---

## 5. Puntos Extra Implementados (Hasta +9 Puntos)

1. **Demostración Interactiva Publicada (+3 pts):**
   * Implementada en [src/reverse_dictionary/web_app.py](file:///Users/osmar/Documents/Uni/Noveno%20Semestre/PLN/DiccionarioInverso/src/reverse_dictionary/web_app.py) con Gradio.
   * Permite ingresar cualquier definición en inglés, seleccionar cualquiera de las 10 estrategias y obtener el top-K con lemas extraídos y tiempos de inferencia en vivo.
2. **Grafo Dirigido y PageRank Personalizado (+2 pts):**
   * Modelado con `nx.DiGraph` preservando la dirección original $estímulo \to respuesta$ y evaluado con `nx.pagerank` reiniciado sobre los lemas de la consulta.
3. **Fusión Híbrida RRF (Grafo + Node2Vec):**
   * Implementada mediante *Reciprocal Rank Fusion* ($k=60$), logrando el mayor $p@1$ (**0.060**) y $MRR$ (**0.088**) de toda la evaluación.
4. **Publicación con Licencia Abierta y Documentación (+2 pts):**
   * Licencia abierta MIT en [LICENSE](file:///Users/osmar/Documents/Uni/Noveno%20Semestre/PLN/DiccionarioInverso/LICENSE), diccionario de datos completo en [data/data_dictionary.md](file:///Users/osmar/Documents/Uni/Noveno%20Semestre/PLN/DiccionarioInverso/data/data_dictionary.md) y código autodocumentado sin comentarios `#`.

---

## 6. Instrucciones de Reproducción con `uv`

El proyecto utiliza exclusivamente `uv` y Python 3.12:

### 6.1 Instalación de dependencias
```bash
uv sync
```

### 6.2 Procesamiento de normas y generación del corpus
```bash
uv run python data/processData.py
uv run python data/build_eval_corpus.py
```

### 6.3 Ejecución del benchmark completo
```bash
uv run python main.py --mode benchmark
```

### 6.4 Lanzamiento de la aplicación web interactiva
```bash
uv run python -m reverse_dictionary.web_app
```
La interfaz estará disponible localmente en `http://127.0.0.1:7860`.

---

## 7. Licencia
Este proyecto se distribuye bajo la licencia MIT. Consulte el archivo [LICENSE](file:///Users/osmar/Documents/Uni/Noveno%20Semestre/PLN/DiccionarioInverso/LICENSE) para más detalles.

