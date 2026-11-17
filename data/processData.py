import argparse
import glob
import logging
import os
import re
from typing import Any

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("process_data")


def normalize_term(term: str) -> str:
    if not term:
        return ""
    normalized = term.strip().lower()
    normalized = normalized.replace("`", "").replace("'", "")
    normalized = re.sub(r"\s*&\s*", "_and_", normalized)
    normalized = re.sub(r"[\s\-]+", "_", normalized)
    return normalized.strip("._")


def parse_usf_raw_files(raw_dir: str) -> tuple[pd.DataFrame, int, list[tuple[str, str, str, int]], list[tuple[str, int, str, str]]]:
    files = sorted(glob.glob(os.path.join(raw_dir, "Cue_Target_Pairs.*")))
    if not files:
        raise FileNotFoundError(f"No se encontraron archivos Cue_Target_Pairs.* en {raw_dir}")

    records: list[dict[str, Any]] = []
    discarded_self_loops: list[tuple[str, str, str, int]] = []
    discarded_empty: list[tuple[str, int, str, str]] = []
    total_raw_lines = 0

    for filepath in files:
        fname = os.path.basename(filepath)
        with open(filepath, "r", encoding="latin-1") as fp:
            for line_idx, line in enumerate(fp, 1):
                clean_line = line.strip()
                if not clean_line or clean_line.startswith("<") or clean_line.startswith("</") or clean_line.startswith("CUE,"):
                    continue

                total_raw_lines += 1
                parts = [p.strip() for p in clean_line.split(",")]
                if len(parts) < 6:
                    discarded_empty.append((fname, line_idx, clean_line, "Menos de 6 columnas"))
                    continue

                raw_cue = parts[0]
                raw_target = parts[1]
                cue_norm = normalize_term(raw_cue)
                target_norm = normalize_term(raw_target)

                if not cue_norm or not target_norm:
                    discarded_empty.append((fname, line_idx, clean_line, "Termino vacio tras normalizacion"))
                    continue

                if cue_norm == target_norm:
                    discarded_self_loops.append((cue_norm, target_norm, fname, line_idx))
                    continue

                try:
                    num_g = int(parts[3])
                except ValueError:
                    num_g = None

                try:
                    num_p = int(parts[4])
                except ValueError:
                    discarded_empty.append((fname, line_idx, clean_line, "Frecuencia no numerica"))
                    continue

                try:
                    fsg = float(parts[5])
                except ValueError:
                    discarded_empty.append((fname, line_idx, clean_line, "FSG no numerico"))
                    continue

                raw_bsg = parts[6] if len(parts) > 6 else ""
                try:
                    bsg = float(raw_bsg)
                except ValueError:
                    bsg = np.nan

                dist_freq = round(1.0 / num_p, 6) if num_p > 0 else 1.0
                dist_strength = round(max(0.0001, 1.0 - fsg), 6)

                records.append({
                    "estimulo": cue_norm,
                    "respuesta": target_norm,
                    "frecuencia": num_p,
                    "tiempo_medio": np.nan,
                    "fuerza": fsg,
                    "fuerza_inversa": bsg,
                    "participantes_grupo": num_g,
                    "dist_frecuencia": dist_freq,
                    "dist_fuerza": dist_strength,
                    "archivo_origen": fname
                })

    return pd.DataFrame(records), total_raw_lines, discarded_self_loops, discarded_empty


def generate_data_dictionary(output_path: str, stats: dict[str, Any]) -> None:
    content = f"""# Diccionario de Datos: Normas de Asociacion Libre USF

## 1. Identificacion y Resumen
- **Conjunto de datos:** University of South Florida (USF) Free Association Norms (Nelson, McEvoy & Schreiber, 2004).
- **Idioma:** Ingles.
- **Pares estimulo-respuesta analizados:** {stats['total_records']:,}
- **Vocabulario de estimulos unicos:** {stats['unique_cues']:,}
- **Vocabulario de respuestas unicas:** {stats['unique_targets']:,}
- **Vocabulario total (nodos del grafo):** {stats['total_vocab']:,}
- **Registros descartados (self-loops):** {stats['self_loops_count']}
- **Registros invalidos descartados:** {stats['empty_count']}

---

## 2. Descripcion de Campos

| Campo | Tipo de Dato | Rango / Valores | Descripcion |
|---|---|---|---|
| `estimulo` | `str` | Lema en minusculas | Palabra estimulo presentada a los participantes (*CUE*). Expresiones multipalabra unidas con guion bajo (`_`). |
| `respuesta` | `str` | Lema en minusculas | Palabra producida como primera respuesta asociada (*TARGET*). |
| `frecuencia` | `int` | $\\ge 1$ | Conteo de participantes que produjeron la respuesta ante el estimulo. |
| `tiempo_medio` | `float` | `NaN` | Tiempo medio de reaccion. No disponible en normas USF originales. Se utilizan distancias de frecuencia y fuerza asociativa. |
| `fuerza` | `float` | $(0.0, 1.0]$ | Fuerza de asociacion directa (*Forward Association Strength*, $FSG$). Proporcion de participantes que dieron la respuesta. |
| `fuerza_inversa` | `float` | $[0.0, 1.0]$ o `NaN` | Fuerza de asociacion inversa (*Backward Association Strength*, $BSG$). |
| `participantes_grupo` | `int` | $\\sim 100$ a 160 | Tamano de la muestra expuesta al estimulo. |
| `dist_frecuencia` | `float` | $(0.0, 1.0]$ | Costo de camino para el grafo calculado como $1 / \\text{{frecuencia}}$. |
| `dist_fuerza` | `float` | $[0.0, 1.0)$ | Costo de camino para el grafo calculado como $1.0 - FSG$. |
| `archivo_origen` | `str` | Nombre de archivo | Archivo original de USF de donde proviene la asociacion. |

---

## 3. Decisiones de Preprocesamiento y Normalizacion
1. **Minusculas y Limpieza:** Todos los terminos convertidos a minusculas, retirando acentos graves y comillas espurias.
2. **Expresiones Multipalabra:** Homologadas mediante guion bajo (`_`).
3. **Simbolo Ampersand:** Reemplazado por `_and_`.
4. **Filtrado de Bucles Propios:** 4 registros donde estimulo equivalia a respuesta fueron descartados.
"""
    with open(output_path, "w", encoding="utf-8") as fp:
        fp.write(content)
    logger.info("Diccionario de datos generado en: %s", output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Procesar normas USF crudas para el Diccionario Inverso.")
    parser.add_argument("--raw_dir", type=str, default="data/usf_raw", help="Directorio con archivos Cue_Target_Pairs.*")
    parser.add_argument("--output_csv", type=str, default="data/usf_norms.csv", help="Ruta del CSV procesado")
    parser.add_argument("--output_dict", type=str, default="data/data_dictionary.md", help="Ruta del diccionario de datos")
    args = parser.parse_args()

    logger.info("Leyendo archivos desde %s", args.raw_dir)
    df, total_raw_lines, self_loops, empty = parse_usf_raw_files(args.raw_dir)

    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    df.to_csv(args.output_csv, index=False, encoding="utf-8")
    logger.info("Normas procesadas guardadas en: %s", args.output_csv)

    unique_cues = int(df["estimulo"].nunique())
    unique_targets = int(df["respuesta"].nunique())
    total_vocab = len(set(df["estimulo"]).union(set(df["respuesta"])))

    stats = {
        "total_records": len(df),
        "unique_cues": unique_cues,
        "unique_targets": unique_targets,
        "total_vocab": total_vocab,
        "self_loops_count": len(self_loops),
        "empty_count": len(empty)
    }

    generate_data_dictionary(args.output_dict, stats)

    logger.info("Lineas crudas leidas: %d", total_raw_lines)
    logger.info("Pares validos exportados: %d", len(df))
    logger.info("Estimulos unicos: %d", unique_cues)
    logger.info("Respuestas unicas: %d", unique_targets)
    logger.info("Vocabulario total (nodos): %d", total_vocab)
    logger.info("Bucles propios descartados: %d", len(self_loops))
    logger.info("Filas invalidas descartadas: %d", len(empty))


if __name__ == "__main__":
    main()
