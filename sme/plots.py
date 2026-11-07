"""Gráficos con matplotlib a partir de las métricas propias."""

import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ._utils import _frame, _is_numeric
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


def plot_pca(data, group=None, scale=True, title="Análisis de componentes principales"):
    """PCA mediante SVD: observaciones, varianza explicada y pesos de atributos.

    data: DataFrame o Dataset. group: nombre de una columna para colorear (admite
    más de dos grupos); nunca se usa como predictor. En Dataset se utiliza la
    clase por defecto y se excluyen también sus categorías declaradas.
    scale: estandarizar con desviación muestral (True) o solo centrar (False).
    Se seleccionan numéricas reales, se omiten constantes y se rechazan ausentes
    e infinitos en predictores. Se necesitan dos filas y dos atributos variables.
    Devuelve un dict con figure, axes, scores, loadings (vectores de la SVD),
    explained_variance (fracciones), center, scale y dropped (constantes).
    No modifica data ni llama a show(). Los índices originales se conservan.
    """
    from .dataset import Dataset

    excluded = set()
    if isinstance(data, Dataset):
        excluded.update(data.categorical)
        if data.class_column is not None:
            excluded.add(data.class_column)
        if group is None:
            group = data.class_column
        data = data.data
    _frame(data, [] if group is None else [group])
    if not isinstance(scale, bool):
        raise ValueError("scale debe ser True o False.")
    if group is not None:
        excluded.add(group)
        if data[group].isna().any():
            raise ValueError("La columna de grupos no puede contener ausentes.")
    columns = [name for name in data if name not in excluded and _is_numeric(data[name])]
    if len(data) < 2 or len(columns) < 2:
        raise ValueError("PCA requiere dos filas y dos atributos numéricos variables.")
    if any(pd.api.types.is_complex_dtype(data[name].dtype) for name in columns):
        raise ValueError("PCA requiere atributos numéricos reales.")
    matrix = data[columns].to_numpy(dtype=float, na_value=np.nan)
    if not np.isfinite(matrix).all():
        raise ValueError("PCA no admite ausentes ni infinitos en los atributos numéricos.")
    variable = np.any(matrix != matrix[0], axis=0)
    dropped = [name for name, keep in zip(columns, variable) if not keep]
    columns = [name for name, keep in zip(columns, variable) if keep]
    if len(columns) < 2:
        raise ValueError("PCA requiere al menos dos atributos numéricos no constantes.")
    matrix = matrix[:, variable]
    center = matrix.mean(axis=0)
    centered = matrix - center
    scales = centered.std(axis=0, ddof=1) if scale else np.ones(len(columns))
    standardized = centered / scales
    _, singular, vt = np.linalg.svd(standardized, full_matrices=False)

    signs = np.where(vt[np.arange(len(vt)), np.abs(vt).argmax(axis=1)] < 0, -1, 1)
    loadings = vt.T * signs
    scores = standardized @ loadings
    explained = singular**2 / np.sum(singular**2)
    components = [f"PC{i + 1}" for i in range(len(singular))]
    scores = pd.DataFrame(scores, index=data.index, columns=components)
    loadings = pd.DataFrame(loadings, index=columns, columns=components)

    fig, axes = plt.subplots(1, 3, figsize=(16, max(4.5, len(columns) * 0.35)),
                             layout="constrained")
    ax = axes[0]
    if group is None:
        ax.scatter(scores.PC1, scores.PC2, color="#237e85", alpha=0.8, s=32)
    else:
        labels = pd.unique(data[group])
        colors = plt.get_cmap("tab10")
        for i, label in enumerate(labels):
            mask = (data[group] == label).to_numpy(dtype=bool)
            ax.scatter(scores.PC1.to_numpy()[mask], scores.PC2.to_numpy()[mask],
                       label=str(label), color=colors(i % 10), alpha=0.8, s=32)
        ax.legend(title=str(group), fontsize=8)
    ax.axhline(0, color="gray", lw=0.6)
    ax.axvline(0, color="gray", lw=0.6)
    ax.set(xlabel=f"PC1 ({explained[0]:.1%})", ylabel=f"PC2 ({explained[1]:.1%})",
           title="Observaciones")
    ax.set_aspect("equal", adjustable="datalim")
    positions = np.arange(len(components))
    axes[1].bar(positions, explained * 100, color="#237e85", label="Individual")
    axes[1].plot(positions, np.cumsum(explained) * 100, "o-", color="#b34747",
                 label="Acumulada")
    axes[1].set_xticks(positions, components, rotation=45)
    axes[1].set(title="Varianza explicada", ylabel="Porcentaje (%)", ylim=(0, 105))
    axes[1].legend(fontsize=8)
    weights = loadings.iloc[:, :2].to_numpy()
    chart = axes[2].imshow(weights, cmap="RdBu", vmin=-1, vmax=1, aspect="auto")
    axes[2].set_xticks([0, 1], ["PC1", "PC2"])
    axes[2].set_yticks(range(len(columns)), columns)
    axes[2].set_title("Pesos de los atributos")
    for i in range(len(columns)):
        for j in range(2):
            axes[2].text(j, i, f"{weights[i, j]:.2f}", ha="center", va="center",
                         color="white" if abs(weights[i, j]) > 0.6 else "black")
    fig.colorbar(chart, ax=axes[2], shrink=0.8, label="Peso (vector unitario)")
    fig.suptitle(title)
    return {"figure": fig, "axes": axes, "scores": scores, "loadings": loadings,
            "explained_variance": pd.Series(explained, index=components),
            "center": pd.Series(center, index=columns),
            "scale": pd.Series(scales, index=columns), "dropped": dropped}
