#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tarea 6 - Inteligencia Artificial en el Procesamiento de Imágenes
Parte III: Implementación práctica de una CNN sencilla para clasificar MNIST.

Universidad Autónoma de Aguascalientes - ICI 7°A
Materia: Procesamiento de Imágenes
Profesor: Hermilo Sánchez Cruz
Alumnos: Hernández Herrera Rodrigo, Huizar Gael, Mata Araiza Maria Fernanda

Uso:
    python cnn_mnist.py                  # entrenamiento con los valores por defecto
    python cnn_mnist.py --epochs 5       # cambiar el número de épocas
    python cnn_mnist.py --help           # ver todas las opciones

Qué hace este script:
    1. Descarga MNIST con torchvision (si no existe ya en ./data).
    2. Separa las 60 000 imágenes de entrenamiento en 55 000 para entrenar y
       5 000 para validar; las 10 000 de prueba solo se usan al final.
    3. Define y entrena una CNN sencilla con PyTorch.
    4. Reporta: arquitectura, número de parámetros, precisión de entrenamiento,
       validación y prueba, matriz de confusión y métricas por clase.
    5. Guarda figuras en ./figuras y resultados numéricos en ./resultados.
"""

import argparse
import csv
import json
import platform
import random
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # backend sin ventana: permite generar figuras en servidores
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

# Colores de las series en las gráficas (azul = entrenamiento, naranja = validación)
COLOR_ENTRENAMIENTO = "#2a78d6"
COLOR_VALIDACION = "#eb6834"
COLOR_TEXTO = "#0b0b0b"
COLOR_TEXTO_SEC = "#52514e"
COLOR_ERROR = "#c0392b"

# Media y desviación estándar de los píxeles de MNIST (valores estándar, calculados
# sobre el conjunto de entrenamiento); se usan para normalizar la entrada.
MNIST_MEDIA = 0.1307
MNIST_DESV = 0.3081


# ---------------------------------------------------------------------------
# 1. Utilidades generales
# ---------------------------------------------------------------------------
def fijar_semilla(semilla: int) -> None:
    """Fija las semillas de todos los generadores aleatorios para que el
    experimento sea reproducible (mismos pesos iniciales, misma partición
    y mismo orden de los lotes)."""
    random.seed(semilla)
    np.random.seed(semilla)
    torch.manual_seed(semilla)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def desnormalizar(imagen: torch.Tensor) -> np.ndarray:
    """Convierte un tensor normalizado (1x28x28) de vuelta a valores [0, 1]
    para poder mostrarlo como imagen."""
    return (imagen.squeeze().numpy() * MNIST_DESV + MNIST_MEDIA).clip(0, 1)


# ---------------------------------------------------------------------------
# 2. Modelo: CNN sencilla
# ---------------------------------------------------------------------------
class CNNSencilla(nn.Module):
    """Red convolucional de dos bloques convolución-ReLU-max pooling seguida
    de un clasificador totalmente conectado.

    Forma de los datos en cada etapa (canales x alto x ancho):
        entrada            1 x 28 x 28
        Conv 3x3 (32)     32 x 28 x 28   (padding=1 conserva el tamaño)
        MaxPool 2x2       32 x 14 x 14
        Conv 3x3 (64)     64 x 14 x 14
        MaxPool 2x2       64 x  7 x  7
        Aplanado               3136
        Densa + ReLU            128
        Dropout (p=0.5)         128
        Densa                    10       (logits, uno por dígito)
    """

    def __init__(self, num_clases: int = 10):
        super().__init__()
        # Bloque 1: extrae rasgos locales simples (bordes, trazos cortos)
        self.bloque1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        # Bloque 2: combina los rasgos anteriores en patrones más complejos
        # (curvas, intersecciones, partes de dígitos)
        self.bloque2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        # Clasificador: convierte el mapa de rasgos en una puntuación por clase
        self.clasificador = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(p=0.5),  # apaga neuronas al azar al entrenar (regularización)
            nn.Linear(128, num_clases),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.bloque1(x)
        x = self.bloque2(x)
        return self.clasificador(x)  # logits; softmax se aplica en la pérdida


def resumen_modelo(modelo: nn.Module, forma_entrada=(1, 1, 28, 28)):
    """Construye una tabla con cada capa, la forma de su salida y su número
    de parámetros, usando 'hooks' que se ejecutan durante una pasada hacia
    adelante con una entrada de prueba."""
    filas, ganchos = [], []

    def registrar(nombre):
        def gancho(modulo, entrada, salida):
            filas.append(
                {
                    "capa": nombre,
                    "tipo": modulo.__class__.__name__,
                    "salida": list(salida.shape[1:]),
                    "parametros": sum(p.numel() for p in modulo.parameters()),
                }
            )

        return gancho

    for nombre, modulo in modelo.named_modules():
        if len(list(modulo.children())) == 0:  # solo capas "hoja"
            ganchos.append(modulo.register_forward_hook(registrar(nombre)))

    modelo.eval()
    dispositivo = next(modelo.parameters()).device
    with torch.no_grad():
        modelo(torch.zeros(forma_entrada, device=dispositivo))
    for g in ganchos:
        g.remove()

    total = sum(p.numel() for p in modelo.parameters())
    entrenables = sum(p.numel() for p in modelo.parameters() if p.requires_grad)
    return filas, total, entrenables


# ---------------------------------------------------------------------------
# 3. Datos
# ---------------------------------------------------------------------------
def cargar_datos(dir_datos: Path, tam_lote: int, semilla: int, n_validacion: int):
    """Descarga (si hace falta) MNIST y crea los DataLoaders de
    entrenamiento, validación y prueba."""
    # ToTensor: convierte la imagen a tensor con valores en [0, 1]
    # Normalize: resta la media y divide entre la desviación estándar
    transformacion = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((MNIST_MEDIA,), (MNIST_DESV,))]
    )
    entrenamiento_completo = datasets.MNIST(
        root=str(dir_datos), train=True, download=True, transform=transformacion
    )
    prueba = datasets.MNIST(
        root=str(dir_datos), train=False, download=True, transform=transformacion
    )

    # Partición reproducible: 55 000 entrenamiento / 5 000 validación
    generador = torch.Generator().manual_seed(semilla)
    n_entrenamiento = len(entrenamiento_completo) - n_validacion
    entrenamiento, validacion = random_split(
        entrenamiento_completo, [n_entrenamiento, n_validacion], generator=generador
    )

    cargadores = {
        # shuffle=True: el orden de las imágenes cambia en cada época
        "entrenamiento": DataLoader(
            entrenamiento, batch_size=tam_lote, shuffle=True,
            generator=torch.Generator().manual_seed(semilla),
        ),
        # Versión sin barajar para evaluar la precisión final de entrenamiento
        "entrenamiento_eval": DataLoader(entrenamiento, batch_size=1000, shuffle=False),
        "validacion": DataLoader(validacion, batch_size=1000, shuffle=False),
        "prueba": DataLoader(prueba, batch_size=1000, shuffle=False),
    }
    return cargadores, prueba


# ---------------------------------------------------------------------------
# 4. Entrenamiento y evaluación
# ---------------------------------------------------------------------------
def entrenar_una_epoca(modelo, cargador, criterio, optimizador, dispositivo):
    """Recorre una vez todo el conjunto de entrenamiento actualizando los pesos.
    Devuelve la pérdida y la precisión promedio de la época (medidas con
    dropout activo, mientras el modelo aprende)."""
    modelo.train()
    perdida_total, aciertos, n = 0.0, 0, 0
    for imagenes, etiquetas in cargador:
        imagenes, etiquetas = imagenes.to(dispositivo), etiquetas.to(dispositivo)

        optimizador.zero_grad()            # 1) reiniciar gradientes
        logits = modelo(imagenes)          # 2) pasada hacia adelante
        perdida = criterio(logits, etiquetas)
        perdida.backward()                 # 3) retropropagación
        optimizador.step()                 # 4) actualización de pesos (Adam)

        perdida_total += perdida.item() * imagenes.size(0)
        aciertos += (logits.argmax(dim=1) == etiquetas).sum().item()
        n += imagenes.size(0)
    return perdida_total / n, aciertos / n


@torch.no_grad()
def evaluar(modelo, cargador, criterio, dispositivo):
    """Evalúa el modelo (sin dropout y sin calcular gradientes).
    Devuelve pérdida, precisión, etiquetas reales, predicciones y
    probabilidades softmax de cada imagen."""
    modelo.eval()
    perdida_total, n = 0.0, 0
    reales, predichas, probabilidades = [], [], []
    for imagenes, etiquetas in cargador:
        imagenes, etiquetas = imagenes.to(dispositivo), etiquetas.to(dispositivo)
        logits = modelo(imagenes)
        perdida_total += criterio(logits, etiquetas).item() * imagenes.size(0)
        prob = torch.softmax(logits, dim=1)
        reales.append(etiquetas.cpu())
        predichas.append(prob.argmax(dim=1).cpu())
        probabilidades.append(prob.cpu())
        n += imagenes.size(0)
    reales = torch.cat(reales).numpy()
    predichas = torch.cat(predichas).numpy()
    probabilidades = torch.cat(probabilidades).numpy()
    return perdida_total / n, float((reales == predichas).mean()), reales, predichas, probabilidades


# ---------------------------------------------------------------------------
# 5. Figuras
# ---------------------------------------------------------------------------
def estilo_graficas():
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.edgecolor": "#b5b4ae",
            "axes.labelcolor": COLOR_TEXTO,
            "axes.titlesize": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.color": COLOR_TEXTO_SEC,
            "ytick.color": COLOR_TEXTO_SEC,
            "axes.grid": True,
            "grid.color": "#e6e5e0",
            "grid.linewidth": 0.8,
            "legend.frameon": False,
            "savefig.bbox": "tight",
            "savefig.dpi": 200,
        }
    )


def guardar(fig, dir_figuras: Path, nombre: str):
    """Guarda cada figura en PNG (para el repositorio) y PDF (para el reporte)."""
    fig.savefig(dir_figuras / f"{nombre}.png")
    fig.savefig(dir_figuras / f"{nombre}.pdf")
    plt.close(fig)


def figura_muestras(conjunto, dir_figuras: Path):
    """Dos ejemplos de cada dígito tomados del conjunto de prueba."""
    etiquetas = conjunto.targets.numpy()
    fig, ejes = plt.subplots(2, 10, figsize=(10, 2.5))
    for digito in range(10):
        indices = np.where(etiquetas == digito)[0][:2]
        for fila, idx in enumerate(indices):
            imagen, _ = conjunto[idx]
            ax = ejes[fila, digito]
            ax.imshow(desnormalizar(imagen), cmap="gray_r")
            ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
            for s in ax.spines.values():
                s.set_visible(False)
            if fila == 0:
                ax.set_title(str(digito), color=COLOR_TEXTO)
    guardar(fig, dir_figuras, "fig_muestras_mnist")


def figura_curvas(historial, dir_figuras: Path):
    """Curvas de pérdida y precisión por época (entrenamiento vs validación)."""
    epocas = [h["epoca"] for h in historial]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.6))

    ax1.plot(epocas, [h["perdida_entrenamiento"] for h in historial], "-o",
             color=COLOR_ENTRENAMIENTO, lw=2, ms=5, label="Entrenamiento")
    ax1.plot(epocas, [h["perdida_validacion"] for h in historial], "-s",
             color=COLOR_VALIDACION, lw=2, ms=5, label="Validación")
    ax1.set_xlabel("Época"); ax1.set_ylabel("Pérdida (entropía cruzada)")
    ax1.set_title("Pérdida por época")
    ax1.legend()

    ax2.plot(epocas, [100 * h["precision_entrenamiento"] for h in historial], "-o",
             color=COLOR_ENTRENAMIENTO, lw=2, ms=5, label="Entrenamiento")
    ax2.plot(epocas, [100 * h["precision_validacion"] for h in historial], "-s",
             color=COLOR_VALIDACION, lw=2, ms=5, label="Validación")
    ax2.set_xlabel("Época"); ax2.set_ylabel("Precisión (%)")
    ax2.set_title("Precisión (accuracy) por época")
    ax2.legend(loc="lower right")

    for ax in (ax1, ax2):
        ax.set_xticks(epocas)
    fig.tight_layout()
    guardar(fig, dir_figuras, "fig_curvas_entrenamiento")


def figura_matriz_confusion(matriz, dir_figuras: Path):
    """Matriz de confusión del conjunto de prueba con el conteo en cada celda.
    Filas = clase real, columnas = clase predicha."""
    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    # Escala logarítmica de color: así los errores (valores pequeños) también
    # se distinguen junto a la diagonal (valores cercanos a 1000).
    from matplotlib.colors import LogNorm

    im = ax.imshow(matriz + 1, cmap="Blues", norm=LogNorm(vmin=1, vmax=matriz.max() + 1))
    ax.grid(False)
    umbral = np.log(matriz.max() + 1) * 0.55
    for i in range(10):
        for j in range(10):
            valor = matriz[i, j]
            if valor == 0:
                continue
            color = "white" if np.log(valor + 1) > umbral else COLOR_TEXTO
            ax.text(j, i, str(valor), ha="center", va="center", fontsize=8,
                    color=color, fontweight="bold" if i == j else "normal")
    ax.set_xticks(range(10)); ax.set_yticks(range(10))
    ax.set_xlabel("Clase predicha"); ax.set_ylabel("Clase real")
    ax.set_title("Matriz de confusión (conjunto de prueba, n = 10 000)")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Número de imágenes (escala log)")
    guardar(fig, dir_figuras, "fig_matriz_confusion")


def figura_ejemplos(conjunto, indices, reales, predichas, probabilidades,
                    titulo, nombre, dir_figuras: Path, es_error: bool):
    """Muestra cinco imágenes con su clase real, la predicha y la confianza."""
    fig, ejes = plt.subplots(1, len(indices), figsize=(10, 2.6))
    for ax, idx in zip(ejes, indices):
        imagen, _ = conjunto[idx]
        ax.imshow(desnormalizar(imagen), cmap="gray_r")
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        for s in ax.spines.values():
            s.set_visible(False)
        conf = 100 * probabilidades[idx, predichas[idx]]
        # Con dos decimales, 99.999 % se mostraría como 100.00 %: se indica "> 99.99 %"
        texto_conf = "> 99.99 %" if conf > 99.99 else f"{conf:.2f} %"
        ax.set_title(f"Real: {reales[idx]}   Pred: {predichas[idx]}\n"
                     f"confianza {texto_conf}\nimagen #{idx}",
                     fontsize=9, color=COLOR_ERROR if es_error else COLOR_TEXTO)
    fig.subplots_adjust(wspace=0.35)
    fig.suptitle(titulo, fontsize=11, color=COLOR_TEXTO, y=1.10)
    guardar(fig, dir_figuras, nombre)


# ---------------------------------------------------------------------------
# 6. Programa principal
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="CNN sencilla para MNIST (PyTorch)")
    parser.add_argument("--epochs", type=int, default=10, help="número de épocas")
    parser.add_argument("--batch-size", type=int, default=128, help="tamaño de lote")
    parser.add_argument("--lr", type=float, default=1e-3, help="tasa de aprendizaje de Adam")
    parser.add_argument("--seed", type=int, default=42, help="semilla aleatoria")
    parser.add_argument("--val-size", type=int, default=5000, help="imágenes para validación")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--fig-dir", type=Path, default=Path("figuras"))
    parser.add_argument("--out-dir", type=Path, default=Path("resultados"))
    args = parser.parse_args()

    fijar_semilla(args.seed)
    estilo_graficas()
    args.fig_dir.mkdir(parents=True, exist_ok=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo: {dispositivo} | PyTorch {torch.__version__}")

    # --- Datos -------------------------------------------------------------
    cargadores, conjunto_prueba = cargar_datos(args.data_dir, args.batch_size,
                                               args.seed, args.val_size)
    print(f"Entrenamiento: {len(cargadores['entrenamiento'].dataset)} | "
          f"Validación: {len(cargadores['validacion'].dataset)} | "
          f"Prueba: {len(conjunto_prueba)}")
    figura_muestras(conjunto_prueba, args.fig_dir)

    # --- Modelo ------------------------------------------------------------
    modelo = CNNSencilla().to(dispositivo)
    filas, total_param, param_entrenables = resumen_modelo(modelo)
    with open(args.out_dir / "arquitectura.txt", "w", encoding="utf-8") as f:
        f.write(str(modelo) + "\n\n")
        f.write(f"{'Capa':<18}{'Tipo':<12}{'Forma de salida':<20}{'Parámetros':>12}\n")
        f.write("-" * 62 + "\n")
        for fila in filas:
            f.write(f"{fila['capa']:<18}{fila['tipo']:<12}{str(fila['salida']):<20}"
                    f"{fila['parametros']:>12,}\n")
        f.write("-" * 62 + "\n")
        f.write(f"Parámetros totales: {total_param:,}\n")
        f.write(f"Parámetros entrenables: {param_entrenables:,}\n")
    print((args.out_dir / "arquitectura.txt").read_text(encoding="utf-8"))

    # Entropía cruzada: combina softmax + log-verosimilitud negativa
    criterio = nn.CrossEntropyLoss()
    optimizador = torch.optim.Adam(modelo.parameters(), lr=args.lr)

    # --- Entrenamiento -----------------------------------------------------
    historial = []
    t_inicio = time.perf_counter()
    for epoca in range(1, args.epochs + 1):
        t0 = time.perf_counter()
        p_ent, a_ent = entrenar_una_epoca(modelo, cargadores["entrenamiento"],
                                          criterio, optimizador, dispositivo)
        p_val, a_val, *_ = evaluar(modelo, cargadores["validacion"], criterio, dispositivo)
        duracion = time.perf_counter() - t0
        historial.append({
            "epoca": epoca,
            "perdida_entrenamiento": p_ent, "precision_entrenamiento": a_ent,
            "perdida_validacion": p_val, "precision_validacion": a_val,
            "segundos": duracion,
        })
        print(f"Época {epoca:2d}/{args.epochs} | pérdida ent {p_ent:.4f} | "
              f"prec ent {100*a_ent:.2f}% | pérdida val {p_val:.4f} | "
              f"prec val {100*a_val:.2f}% | {duracion:.1f} s")
    tiempo_entrenamiento = time.perf_counter() - t_inicio

    with open(args.out_dir / "historial.csv", "w", newline="", encoding="utf-8") as f:
        escritor = csv.DictWriter(f, fieldnames=list(historial[0].keys()))
        escritor.writeheader()
        escritor.writerows(historial)
    torch.save(modelo.state_dict(), args.out_dir / "modelo_cnn_mnist.pt")
    figura_curvas(historial, args.fig_dir)

    # --- Evaluación final (modo evaluación, sin dropout) -------------------
    p_ent, a_ent, *_ = evaluar(modelo, cargadores["entrenamiento_eval"], criterio, dispositivo)
    p_val, a_val, *_ = evaluar(modelo, cargadores["validacion"], criterio, dispositivo)
    t0 = time.perf_counter()
    p_pru, a_pru, reales, predichas, probs = evaluar(modelo, cargadores["prueba"],
                                                     criterio, dispositivo)
    tiempo_inferencia = time.perf_counter() - t0

    print(f"\nPrecisión de entrenamiento: {100*a_ent:.2f}%  (pérdida {p_ent:.4f})")
    print(f"Precisión de validación:    {100*a_val:.2f}%  (pérdida {p_val:.4f})")
    print(f"Precisión de prueba:        {100*a_pru:.2f}%  (pérdida {p_pru:.4f})")

    # Matriz de confusión y métricas por clase (precision, recall, F1)
    matriz = confusion_matrix(reales, predichas, labels=list(range(10)))
    figura_matriz_confusion(matriz, args.fig_dir)
    reporte = classification_report(reales, predichas, digits=4)
    print("\n" + reporte)
    (args.out_dir / "reporte_clasificacion.txt").write_text(reporte, encoding="utf-8")
    prec, rec, f1, soporte = precision_recall_fscore_support(reales, predichas,
                                                             labels=list(range(10)))

    # Pares (real -> predicha) que más se confunden
    fuera_diagonal = matriz.copy()
    np.fill_diagonal(fuera_diagonal, 0)
    pares = sorted(((int(fuera_diagonal[i, j]), i, j) for i in range(10) for j in range(10)
                    if fuera_diagonal[i, j] > 0), reverse=True)

    # --- Ejemplos correctos e incorrectos ------------------------------------
    rng = np.random.default_rng(args.seed)
    idx_correctos = np.where(reales == predichas)[0]
    idx_errores = np.where(reales != predichas)[0]
    # Cinco aciertos al azar, cada uno de un dígito distinto
    clases = rng.choice(10, size=5, replace=False)
    ejemplos_correctos = [int(rng.choice(idx_correctos[reales[idx_correctos] == c]))
                          for c in sorted(clases)]
    # Los cinco errores en los que el modelo estaba más seguro de su respuesta
    confianza_errores = probs[idx_errores, predichas[idx_errores]]
    ejemplos_errores = [int(i) for i in idx_errores[np.argsort(-confianza_errores)[:5]]]

    figura_ejemplos(conjunto_prueba, ejemplos_correctos, reales, predichas, probs,
                    "Ejemplos clasificados correctamente", "fig_ejemplos_correctos",
                    args.fig_dir, es_error=False)
    figura_ejemplos(conjunto_prueba, ejemplos_errores, reales, predichas, probs,
                    "Ejemplos clasificados incorrectamente (errores con mayor confianza)",
                    "fig_ejemplos_incorrectos", args.fig_dir, es_error=True)

    # --- Guardar todos los resultados en JSON --------------------------------
    resultados = {
        "configuracion": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "entorno": {
            "pytorch": torch.__version__, "python": platform.python_version(),
            "dispositivo": str(dispositivo), "hilos_cpu": torch.get_num_threads(),
            "procesador": platform.processor() or platform.machine(),
        },
        "parametros_totales": total_param,
        "parametros_entrenables": param_entrenables,
        "capas": filas,
        "precision_entrenamiento": a_ent, "perdida_entrenamiento": p_ent,
        "precision_validacion": a_val, "perdida_validacion": p_val,
        "precision_prueba": a_pru, "perdida_prueba": p_pru,
        "errores_prueba": int(len(idx_errores)),
        "tiempo_entrenamiento_s": tiempo_entrenamiento,
        "tiempo_inferencia_prueba_s": tiempo_inferencia,
        "por_clase": [
            {"clase": c, "precision": float(prec[c]), "recall": float(rec[c]),
             "f1": float(f1[c]), "soporte": int(soporte[c])} for c in range(10)
        ],
        "macro": {"precision": float(prec.mean()), "recall": float(rec.mean()),
                  "f1": float(f1.mean())},
        "matriz_confusion": matriz.tolist(),
        "pares_mas_confundidos": [{"real": i, "predicha": j, "conteo": n}
                                  for n, i, j in pares[:10]],
        "ejemplos_correctos": [{"indice": i, "real": int(reales[i]), "predicha": int(predichas[i]),
                                "confianza": float(probs[i, predichas[i]])}
                               for i in ejemplos_correctos],
        "ejemplos_incorrectos": [{"indice": i, "real": int(reales[i]),
                                  "predicha": int(predichas[i]),
                                  "confianza": float(probs[i, predichas[i]]),
                                  "prob_clase_real": float(probs[i, reales[i]])}
                                 for i in ejemplos_errores],
        "historial": historial,
    }
    with open(args.out_dir / "metricas.json", "w", encoding="utf-8") as f:
        json.dump(resultados, f, indent=2, ensure_ascii=False)
    print(f"\nFiguras guardadas en: {args.fig_dir.resolve()}")
    print(f"Resultados guardados en: {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()
