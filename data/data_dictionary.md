# Diccionario de Datos: Normas de Asociacion Libre USF

## 1. Identificacion y Resumen
- **Conjunto de datos:** University of South Florida (USF) Free Association Norms (Nelson, McEvoy & Schreiber, 2004).
- **Idioma:** Ingles.
- **Pares estimulo-respuesta analizados:** 72,151
- **Vocabulario de estimulos unicos:** 5,017
- **Vocabulario de respuestas unicas:** 10,450
- **Vocabulario total (nodos del grafo):** 10,599
- **Registros descartados (self-loops):** 4
- **Registros invalidos descartados:** 0

---

## 2. Descripcion de Campos

| Campo | Tipo de Dato | Rango / Valores | Descripcion |
|---|---|---|---|
| `estimulo` | `str` | Lema en minusculas | Palabra estimulo presentada a los participantes (*CUE*). Expresiones multipalabra unidas con guion bajo (`_`). |
| `respuesta` | `str` | Lema en minusculas | Palabra producida como primera respuesta asociada (*TARGET*). |
| `frecuencia` | `int` | $\ge 1$ | Conteo de participantes que produjeron la respuesta ante el estimulo. |
| `tiempo_medio` | `float` | `NaN` | Tiempo medio de reaccion. No disponible en normas USF originales. Se utilizan distancias de frecuencia y fuerza asociativa. |
| `fuerza` | `float` | $(0.0, 1.0]$ | Fuerza de asociacion directa (*Forward Association Strength*, $FSG$). Proporcion de participantes que dieron la respuesta. |
| `fuerza_inversa` | `float` | $[0.0, 1.0]$ o `NaN` | Fuerza de asociacion inversa (*Backward Association Strength*, $BSG$). |
| `participantes_grupo` | `int` | $\sim 100$ a 160 | Tamano de la muestra expuesta al estimulo. |
| `dist_frecuencia` | `float` | $(0.0, 1.0]$ | Costo de camino para el grafo calculado como $1 / \text{frecuencia}$. |
| `dist_fuerza` | `float` | $[0.0, 1.0)$ | Costo de camino para el grafo calculado como $1.0 - FSG$. |
| `archivo_origen` | `str` | Nombre de archivo | Archivo original de USF de donde proviene la asociacion. |

---

## 3. Decisiones de Preprocesamiento y Normalizacion
1. **Minusculas y Limpieza:** Todos los terminos convertidos a minusculas, retirando acentos graves y comillas espurias.
2. **Expresiones Multipalabra:** Homologadas mediante guion bajo (`_`).
3. **Simbolo Ampersand:** Reemplazado por `_and_`.
4. **Filtrado de Bucles Propios:** 4 registros donde estimulo equivalia a respuesta fueron descartados.


---

## 4. Conjunto de Evaluacion: Definiciones en Lenguaje Natural

- **Archivo:** `data/evaluation_corpus.csv`
- **Fuente primaria:** WordNet (nltk.corpus.wordnet).
- **Total de definiciones evaluadas:** 400
- **Particion de desarrollo (`dev`):** 100 definiciones (destinadas a la exploracion y calibracion de hiperparametros).
- **Particion de prueba (`test`):** 300 definiciones (destinadas a la evaluacion comparativa final e informe de resultados).

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
