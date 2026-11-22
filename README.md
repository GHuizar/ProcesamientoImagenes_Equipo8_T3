# Equipo 8 - Tarea 3 — Inteligencia Artificial en el Procesamiento de Imágenes

**Universidad Autónoma de Aguascalientes** · ICI 7°A
**Materia:** Procesamiento de Imágenes · **Profesor:** Hermilo Sánchez Cruz

**Alumnos:**
- Hernández Herrera Rodrigo
- Huizar Gael
- Mata Araiza Maria Fernanda

Implementación de una red neuronal convolucional (CNN) sencilla en **PyTorch** para clasificar los dígitos escritos a mano de **MNIST**, y comparación con dos métodos clásicos (k-NN sobre píxeles y HOG + SVM lineal).

## Resultados

| Método | Exactitud en prueba | Errores (de 10,000) | Parámetros |
|---|---|---|---|
| k-NN (k = 3), píxeles crudos | 96.92 % | 308 | — |
| HOG + SVM lineal | 97.23 % | 277 | 3,250 |
| **CNN (este trabajo)** | **99.37 %** | **63** | 421,642 |

- Exactitud de entrenamiento de la CNN: 99.82 % (55,000 imágenes)
- Exactitud de validación: 99.18 % (5,000 imágenes)
- Exactitud de prueba: 99.37 % (10,000 imágenes)
- Entrenamiento: 10 épocas, Adam (lr = 0.001), lotes de 128, en CPU (≈ 4 min)

## Estructura

```
├── cnn_mnist.py            # Parte III: datos, CNN, entrenamiento, evaluación y figuras
├── baseline_clasico.py     # Comparación con métodos clásicos (k-NN y HOG + SVM)
├── requirements.txt
├── figuras/                # Figuras generadas (PNG y PDF)
│   ├── fig_muestras_mnist
│   ├── fig_curvas_entrenamiento
│   ├── fig_matriz_confusion
│   ├── fig_ejemplos_correctos
│   └── fig_ejemplos_incorrectos
├── resultados/
│   ├── metricas.json             # todas las métricas (incluye matriz de confusión)
│   ├── historial.csv             # pérdida y exactitud por época
│   ├── arquitectura.txt          # resumen de capas y número de parámetros
│   ├── reporte_clasificacion.txt # precisión, recall y F1 por clase
│   ├── baseline_clasico.json     # resultados de los métodos clásicos
│   └── modelo_cnn_mnist.pt       # pesos entrenados
└── reporte/
    ├── reporte_tarea6.pdf        # reporte final
    ├── reporte_tarea6.tex        # fuente LaTeX del reporte
    └── generar_macros.py         # pasa las cifras de resultados/ al reporte
```

## Cómo ejecutarlo

```bash
pip install -r requirements.txt
python cnn_mnist.py          # descarga MNIST en ./data, entrena y genera figuras/ y resultados/
python baseline_clasico.py   # entrena k-NN y HOG + SVM con la misma partición
```

Opciones útiles de `cnn_mnist.py`: `--epochs`, `--batch-size`, `--lr`, `--seed` (ver `python cnn_mnist.py --help`).
Con la semilla por defecto (42) los resultados en CPU son reproducibles.

## Compilar el reporte

```bash
cd reporte
python generar_macros.py     # genera resultados.tex a partir de ../resultados/*.json
xelatex reporte_tarea6.tex   # ejecutar dos veces para actualizar índice y referencias
```
