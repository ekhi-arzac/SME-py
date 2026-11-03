"""Gráficos con matplotlib a partir de las métricas propias."""

import math
import matplotlib.pyplot as plt

from .metrics import auc, roc_curve


def plot_roc(scores, classes, positive=None, ax=None, title="Curva ROC"):
    """Dibujar ROC y AUC; entradas de datos como roc_curve().

    ax: ejes matplotlib opcionales; title: título. Devuelve el objeto Axes,
    sin llamar a show(), para permitir mostrar o guardar la figura externamente.
    """

    # Materializar por si el usuario pasa generadores que solo se recorren una vez.
    scores, classes = list(scores), list(classes)
    curve = roc_curve(scores, classes, positive)
    area = auc(scores, classes, positive)
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 5))
    ax.plot(curve["fpr"], curve["tpr"], color="#237e85", lw=2, label=f"AUC = {area:.3f}")
    ax.plot([0, 1], [0, 1], "--", color="gray")
    ax.set(xlabel="Tasa de falsos positivos", ylabel="Tasa de verdaderos positivos",
           title=title, xlim=(0, 1), ylim=(0, 1))
    ax.set_aspect("equal")
    ax.legend(loc="lower right")
    return ax


def plot_association(association, title="Asociaciones"):
    """Mostrar Pearson y MI en paneles separados, con valores y barras de color.

    association: dict de association_matrix(). title: título de figura.
    Devuelve (Figure, array de Axes). Pearson usa [-1,1]; MI, [0,máximo] bits.
    Celdas sin datos o correspondientes al otro método se muestran en gris.
    """

    values, methods = association["values"], association["methods"]
    if values.empty or values.shape != methods.shape:
        raise ValueError("Se requiere una matriz de asociación no vacía.")
    n = len(values)
    fig, axes = plt.subplots(1, 2, figsize=(max(10, n * 2.3), max(4, n * 0.8)), layout="constrained")
    for kind, ax in zip(("pearson", "mi"), axes):
        matrix = []
        for i in range(n):
            row = []
            for j in range(n):
                valid = (methods.iloc[i, j] == "pearson") == (kind == "pearson")
                row.append(float(values.iloc[i, j]) if valid else math.nan)
            matrix.append(row)
        observed = [v for row in matrix for v in row if not math.isnan(v)]
        upper = 1 if kind == "pearson" else max([1] + observed)
        cmap = plt.get_cmap("RdBu" if kind == "pearson" else "YlGnBu").copy()
        cmap.set_bad("#e6e6e6")
        chart = ax.imshow(matrix, vmin=-1 if kind == "pearson" else 0, vmax=upper, cmap=cmap)
        ax.set_xticks(range(n), values.columns, rotation=55, ha="right")
        ax.set_yticks(range(n), values.index)
        ax.set_title("Pearson" if kind == "pearson" else "Información mutua (bits)")
        for i, row in enumerate(matrix):
            for j, value in enumerate(row):
                if not math.isnan(value):
                    ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=9,
                            bbox={"facecolor": "white", "alpha": 0.7, "edgecolor": "none"})
        fig.colorbar(chart, ax=ax, shrink=0.75)
    fig.suptitle(title)
    return fig, axes
