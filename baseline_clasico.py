#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Equipo 8 - Tarea 3 - Comparación con métodos clásicos (complemento de la Parte II).

Entrena dos clasificadores "clásicos" sobre exactamente las mismas 55 000
imágenes de entrenamiento que la CNN y los evalúa en las mismas 10 000
imágenes de prueba:

    a) k vecinos más cercanos (k-NN, k = 3) sobre los píxeles crudos.
    b) Descriptor HOG (histograma de gradientes orientados, rasgo diseñado
       a mano) + máquina de soporte vectorial (SVM) lineal.

Uso:
    python baseline_clasico.py
(requiere haber descargado MNIST antes, p. ej. ejecutando cnn_mnist.py)
"""

import json
import time
from pathlib import Path

import numpy as np
import torch
from skimage.feature import hog
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from torch.utils.data import random_split
from torchvision import datasets

SEMILLA = 42
N_VALIDACION = 5000


def cargar_mnist(dir_datos=Path("data")):
    """Devuelve las imágenes como arreglos uint8 (N x 28 x 28) y las etiquetas.
    Usa la misma partición 55 000 / 5 000 que cnn_mnist.py (misma semilla)."""
    entrenamiento = datasets.MNIST(str(dir_datos), train=True, download=True)
    prueba = datasets.MNIST(str(dir_datos), train=False, download=True)
    generador = torch.Generator().manual_seed(SEMILLA)
    sub_ent, _ = random_split(entrenamiento,
                              [len(entrenamiento) - N_VALIDACION, N_VALIDACION],
                              generator=generador)
    idx = np.array(sub_ent.indices)
    x_ent = entrenamiento.data.numpy()[idx]
    y_ent = entrenamiento.targets.numpy()[idx]
    return x_ent, y_ent, prueba.data.numpy(), prueba.targets.numpy()


def rasgos_hog(imagenes):
    """Calcula el descriptor HOG de cada imagen: celdas de 7x7 píxeles,
    bloques de 2x2 celdas y 9 orientaciones -> vector de 324 valores."""
    return np.array([
        hog(img, orientations=9, pixels_per_cell=(7, 7), cells_per_block=(2, 2),
            block_norm="L2-Hys")
        for img in imagenes
    ], dtype=np.float32)


def main():
    x_ent, y_ent, x_pru, y_pru = cargar_mnist()
    resultados = {}

    # --- a) k-NN sobre píxeles crudos (sin entrenamiento propiamente dicho:
    #        solo memoriza los ejemplos; el costo está en la predicción) -------
    t0 = time.perf_counter()
    knn = KNeighborsClassifier(n_neighbors=3, n_jobs=-1)
    knn.fit(x_ent.reshape(len(x_ent), -1) / 255.0, y_ent)
    t_ajuste = time.perf_counter() - t0
    t0 = time.perf_counter()
    pred = knn.predict(x_pru.reshape(len(x_pru), -1) / 255.0)
    t_pred = time.perf_counter() - t0
    resultados["knn_pixeles"] = {
        "precision_prueba": float((pred == y_pru).mean()),
        "tiempo_ajuste_s": t_ajuste, "tiempo_prediccion_s": t_pred,
        "dimension_rasgos": 784,
    }
    print(f"k-NN (k=3) píxeles: {100*(pred == y_pru).mean():.2f}%  "
          f"(ajuste {t_ajuste:.1f} s, predicción {t_pred:.1f} s)")

    # --- b) HOG + SVM lineal ----------------------------------------------
    t0 = time.perf_counter()
    h_ent, h_pru = rasgos_hog(x_ent), rasgos_hog(x_pru)
    t_hog = time.perf_counter() - t0
    escalador = StandardScaler().fit(h_ent)
    t0 = time.perf_counter()
    svm = LinearSVC(C=1.0, max_iter=5000, random_state=SEMILLA)
    svm.fit(escalador.transform(h_ent), y_ent)
    t_ajuste = time.perf_counter() - t0
    t0 = time.perf_counter()
    pred = svm.predict(escalador.transform(h_pru))
    t_pred = time.perf_counter() - t0
    resultados["hog_svm_lineal"] = {
        "precision_prueba": float((pred == y_pru).mean()),
        "tiempo_extraccion_hog_s": t_hog, "tiempo_ajuste_s": t_ajuste,
        "tiempo_prediccion_s": t_pred, "dimension_rasgos": int(h_ent.shape[1]),
        "parametros": int(svm.coef_.size + svm.intercept_.size),
    }
    print(f"HOG + SVM lineal:   {100*(pred == y_pru).mean():.2f}%  "
          f"(HOG {t_hog:.1f} s, ajuste {t_ajuste:.1f} s, predicción {t_pred:.2f} s)")

    Path("resultados").mkdir(exist_ok=True)
    with open("resultados/baseline_clasico.json", "w", encoding="utf-8") as f:
        json.dump(resultados, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
