#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Lee los resultados del experimento (../resultados/*.json) y escribe
resultados.tex con macros de LaTeX, para que todas las cifras del reporte
salgan directamente de los archivos generados por el código."""

import json
from pathlib import Path

AQUI = Path(__file__).resolve().parent
m = json.loads((AQUI.parent / "resultados" / "metricas.json").read_text(encoding="utf-8"))
b = json.loads((AQUI.parent / "resultados" / "baseline_clasico.json").read_text(encoding="utf-8"))


def miles(n: int) -> str:
    """421642 -> 421,642 (separador de miles usado en México)."""
    return f"{n:,}".replace(",", "{,}")


def pct(x: float, d: int = 2) -> str:
    return f"{100 * x:.{d}f}"


lineas = []


def macro(nombre, valor):
    lineas.append(f"\\newcommand{{\\{nombre}}}{{{valor}}}")


macro("ParamTotal", miles(m["parametros_totales"]))
macro("AccTrain", pct(m["precision_entrenamiento"]))
macro("AccVal", pct(m["precision_validacion"]))
macro("AccTest", pct(m["precision_prueba"]))
macro("ErrTest", pct(1 - m["precision_prueba"]))
macro("LossTrain", f"{m['perdida_entrenamiento']:.4f}")
macro("LossVal", f"{m['perdida_validacion']:.4f}")
macro("LossTest", f"{m['perdida_prueba']:.4f}")
macro("NErroresTest", str(m["errores_prueba"]))
macro("NAciertosTest", miles(10000 - m["errores_prueba"]))
macro("Epocas", str(m["configuracion"]["epochs"]))
macro("TamLote", str(m["configuracion"]["batch_size"]))
macro("TiempoEnt", f"{m['tiempo_entrenamiento_s']:.0f}")
macro("TiempoEntMin", f"{m['tiempo_entrenamiento_s'] / 60:.1f}")
macro("TiempoEpoca", f"{m['tiempo_entrenamiento_s'] / m['configuracion']['epochs']:.1f}")
macro("TiempoInf", f"{m['tiempo_inferencia_prueba_s']:.1f}")
macro("TiempoInfImg", f"{1000 * m['tiempo_inferencia_prueba_s'] / 10000:.2f}")
macro("MacroP", pct(m["macro"]["precision"]))
macro("MacroR", pct(m["macro"]["recall"]))
macro("MacroF", pct(m["macro"]["f1"]))
macro("VersionTorch", m["entorno"]["pytorch"].split("+")[0])
macro("VersionPython", m["entorno"]["python"])

# Historial por época (filas de tabla)
filas = []
for h in m["historial"]:
    filas.append(
        f"{h['epoca']} & {h['perdida_entrenamiento']:.4f} & {pct(h['precision_entrenamiento'])} & "
        f"{h['perdida_validacion']:.4f} & {pct(h['precision_validacion'])} & {h['segundos']:.1f} \\\\"
    )
macro("FilasHistorial", "\n".join(filas))

# Métricas por clase (filas de tabla)
filas = []
for c in m["por_clase"]:
    filas.append(
        f"{c['clase']} & {pct(c['precision'])} & {pct(c['recall'])} & {pct(c['f1'])} & {miles(c['soporte'])} \\\\"
    )
macro("FilasPorClase", "\n".join(filas))

# Clase con menor y mayor F1
peor = min(m["por_clase"], key=lambda c: c["f1"])
mejor = max(m["por_clase"], key=lambda c: c["f1"])
macro("PeorClase", str(peor["clase"]))
macro("PeorClaseF", pct(peor["f1"]))
macro("PeorClaseR", pct(peor["recall"]))
macro("MejorClase", str(mejor["clase"]))
macro("MejorClaseF", pct(mejor["f1"]))

# Errores totales por clase real (fila de la matriz sin la diagonal)
mat = m["matriz_confusion"]
errores_real = {i: sum(mat[i]) - mat[i][i] for i in range(10)}
macro("ErroresNueve", str(errores_real[9]))
macro("ErroresCinco", str(errores_real[5]))
macro("ErroresUno", str(errores_real[1]))
macro("ErroresCero", str(errores_real[0]))

# Pares más confundidos
for k, p in enumerate(m["pares_mas_confundidos"][:5], start=1):
    letra = "ABCDE"[k - 1]
    macro(f"Par{letra}", f"{p['real']}$\\to${p['predicha']}")
    macro(f"Par{letra}N", str(p["conteo"]))

# Ejemplos incorrectos
for k, e in enumerate(m["ejemplos_incorrectos"], start=1):
    letra = "ABCDE"[k - 1]
    macro(f"Err{letra}Idx", str(e["indice"]))
    macro(f"Err{letra}Real", str(e["real"]))
    macro(f"Err{letra}Pred", str(e["predicha"]))
    macro(f"Err{letra}Conf", pct(e["confianza"], 1))

# Métodos clásicos
macro("KnnAcc", pct(b["knn_pixeles"]["precision_prueba"]))
macro("KnnPred", f"{b['knn_pixeles']['tiempo_prediccion_s']:.1f}")
macro("KnnAjuste", f"{b['knn_pixeles']['tiempo_ajuste_s']:.1f}")
macro("HogAcc", pct(b["hog_svm_lineal"]["precision_prueba"]))
macro("HogAjuste", f"{b['hog_svm_lineal']['tiempo_ajuste_s']:.0f}")
macro("HogExtr", f"{b['hog_svm_lineal']['tiempo_extraccion_hog_s']:.0f}")
macro("HogPred", f"{b['hog_svm_lineal']['tiempo_prediccion_s']:.2f}")
macro("HogParam", miles(b["hog_svm_lineal"]["parametros"]))
macro("HogDim", str(b["hog_svm_lineal"]["dimension_rasgos"]))
macro("HogErrores", str(round((1 - b["hog_svm_lineal"]["precision_prueba"]) * 10000)))
macro("KnnErrores", str(round((1 - b["knn_pixeles"]["precision_prueba"]) * 10000)))

(AQUI / "resultados.tex").write_text(
    "% Archivo generado automáticamente por generar_macros.py - no editar a mano\n"
    + "\n".join(lineas) + "\n",
    encoding="utf-8",
)
print("resultados.tex generado con", len(lineas), "macros")
