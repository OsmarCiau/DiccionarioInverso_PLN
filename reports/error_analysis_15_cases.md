# Analisis de Casos de Error

| No. | Concepto Objetivo | Definicion de Entrada | Lemas Extraidos | Top-5 Candidatos Predichos | Causa del Error |
|---|---|---|---|---|---|
| 1 | **tribe** | *a social division of (usually preliterate) people* | `social, division, usually, people` | `crowd, group, population, society, person` | sesgo_asociativo_o_polisemia |
| 2 | **drench** | *cover with liquid; pour liquid onto* | `cover, liquid, pour` | `water, rain, seal, umbrella, valve` | sesgo_asociativo_o_polisemia |
| 3 | **steer** | *an indication of potential opportunity* | `potential, opportunity` | `chance, possibility, great, future, lack` | descripcion_ambigua_o_corta |
| 4 | **halo** | *an indication of radiant light drawn around the head of a saint* | `light, draw, head, saint` | `bulb, lamp, hat, white, idea` | sesgo_asociativo_o_polisemia |
| 5 | **service** | *work done by one person or group that benefits another* | `work, person, group, benefit` | `job, labor, staff, people, place` | sesgo_asociativo_o_polisemia |
| 6 | **poise** | *a cgs unit of dynamic viscosity equal to one dyne-second per square centimeter; the viscosity of a fluid in which a force of one dyne per square centimeter maintains a velocity of 1 centimeter per second* | `unit, equal, second, centimeter, the, one, square, velocity` | `two, number, math, digit, percent` | sesgo_asociativo_o_polisemia |
| 7 | **after** | *happening at a time subsequent to a reference time* | `happen, time, reference` | `clock, moment, watch, era, when` | sesgo_asociativo_o_polisemia |
| 8 | **crooked** | *bend or cause to bend* | `bend, cause` | `make, break, accident, good, dent` | descripcion_ambigua_o_corta |
| 9 | **film** | *a form of entertainment that enacts a story by sound and a sequence of images giving the illusion of continuous movement* | `form, entertainment, story, sound, sequence, image, give, illusion, continuous, movement` | `book, money, issue, music, time` | sesgo_asociativo_o_polisemia |
| 10 | **growth** | *(biology) the process of an individual organism growing organically; a purely biological unfolding of events involved in an organism changing gradually from a simple to a more complex level* | `biology, the, process, an, grow, of, event, involve, organism, from, simple, more` | `work, big, small, money, time` | sesgo_asociativo_o_polisemia |
| 11 | **thermometer** | *measuring instrument for measuring temperature* | `measure, instrument, temperature` | `gauge, scales, scale, tape, ruler` | sesgo_asociativo_o_polisemia |
| 12 | **linen** | *a fabric woven with fibers from the flax plant* | `fabric, weave, fiber, plant` | `flower, sow, seed, garden, tree` | sesgo_asociativo_o_polisemia |
| 13 | **assistance** | *the activity of contributing to the fulfillment of a need or furtherance of an effort or purpose* | `activity, fulfillment, need, effort, purpose` | `work, money, strive, want, job` | sesgo_asociativo_o_polisemia |
| 14 | **romantic** | *expressive of or exciting sexual love or romance* | `exciting, sexual, love, romance` | `affection, adore, hate, like, passion` | recuperacion_suboptima |
| 15 | **bathtub** | *a relatively large open container that you fill with water and use to wash the body* | `large, open, container, fill, water, use, wash, body` | `faucet, flood, splash, fountain, ocean` | sesgo_asociativo_o_polisemia |

---

## Discusion y Respuestas a Preguntas Guia

### 1. ¿Que informacion se pierde al modelar las asociaciones, que son dirigidas, con un grafo no dirigido?
En las normas de asociacion libre, la asimetria cognitiva es fundamental: un estimulo $A$ evoca una respuesta $B$ con mucha mayor probabilidad que la inversa (por ejemplo, `zebra` $\to$ `stripes` con alta fuerza, pero `stripes` suele evocar `lines`, `candy` o `tiger`, rara vez `zebra`). Al colapsar el grafo en no dirigido, una palabra muy general con miles de respuestas entrantes puede "inundar" artificialmente los caminos mas cortos hacia conceptos especificos. Al evaluar el **PageRank**, se preserva el flujo direccional genuino de la asociacion humana, evitando falsos atajos semanticos.

### 2. ¿Por que la cercania en el grafo reducido mejora p@5 respecto a la intermediacion, pero no necesariamente p@1?
La centralidad de cercania suma las distancias hacia todos los lemas de la definicion. Esto favorece fuertemente a candidatos situados en el «centro de gravedad» de los conceptos de entrada, logrando que el concepto correcto casi siempre caiga dentro del vecindario top 5 ($p@5$). Por el contrario, la intermediacion (*betweenness*) exige que el nodo este exactamente en el camino critico entre pares de lemas: cuando un lema clave conecta de forma unica y directa con la palabra objetivo, la intermediacion puede asignarle el rango 1 ($p@1$), pero penaliza a conceptos que estan equidistantemente cercanos sin ser cuellos de botella topologicos.

### 3. ¿Que ponderacion funciona mejor y por que? ¿Como influye la transformacion de similitud a distancia?
La ponderacion de **Fuerza de Asociacion** ($1.0 - FSG$) supera consistentemente a la frecuencia simple ($1/f$). La fuerza de asociacion es una probabilidad condicional normalizada ($P(\text{respuesta}|\text{estimulo}) = \#P / \#G$) que desacopla el tamano de muestra del experimento, mientras que la frecuencia cruda es sensible a variaciones en el numero de participantes. La transformacion lineal $1.0 - FSG$ acota los costos de transicion entre $0$ y $1$, permitiendo que Dijkstra compute sumas de penalizacion proporcionadas a la distancia cognitiva.
