"""Estadísticos y métricas propios con contadores y operaciones básicas."""

import math
from collections import Counter

import pandas as pd

from ._utils import _class, _frame, _is_numeric, _labels, _numeric, _pairs


def mean(x, na_rm=True):
    """Media propia de un iterable numérico; float o NaN sin observaciones.

    na_rm=True omite ausentes; False devuelve NaN si encuentra alguno.
    Rechaza infinitos y datos no numéricos.
    """
    values = _numeric(x)
    if not na_rm and any(v is None for v in values):
        return math.nan
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else math.nan


def variance(x):
    """Varianza muestral propia de un iterable numérico, omitiendo ausentes.

    Devuelve sum((x-media)**2)/(n-1), o NaN con menos de dos observaciones.
    """
    values = [v for v in _numeric(x) if v is not None]
    if len(values) < 2:
        return math.nan
    center = mean(values)
    return sum((v - center) ** 2 for v in values) / (len(values) - 1)


def std_dev(x):
    """Raíz de la varianza muestral de un iterable numérico; NaN si n < 2."""
    return math.sqrt(variance(x))


def entropy(x):
    """Entropía Shannon en bits de un iterable de categorías hashables.

    Omite ausentes; devuelve -sum(p*log2(p)), o NaN sin observaciones.
    """
    counts = Counter(v for v in x if not pd.isna(v))
    n = sum(counts.values())
    return -sum((count / n) * math.log2(count / n) for count in counts.values()) if n else math.nan


def cross_entropy(x, y):
    """Entropía cruzada H(P,Q) en bits, con x e y muestras de categorías.

    Omite ausentes por separado y alinea por etiqueta, no por posición.
    Devuelve Inf si falta en Q una categoría de P; NaN si alguna muestra está vacía.
    Las muestras pueden tener distinta longitud. No es una entropía conjunta.
    """
    p = Counter(v for v in x if not pd.isna(v))
    q = Counter(v for v in y if not pd.isna(v))
    n, m = sum(p.values()), sum(q.values())
    if not n or not m:
        return math.nan
    if any(q[label] == 0 for label in p):
        return math.inf
    return -sum(count / n * math.log2(q[label] / m) for label, count in p.items())


def mutual_info(x, y):
    """Información mutua empírica en bits de categorías de igual longitud.

    Omite pares con ausentes. Devuelve sum(pxy*log2(pxy/(px*py))),
    o NaN sin pares válidos. Solo recorre combinaciones observadas.
    """
    pairs = _pairs(x, y)
    n = len(pairs)
    if not n:
        return math.nan
    joint = Counter(pairs)
    px = Counter(a for a, _ in pairs)
    py = Counter(b for _, b in pairs)
    return max(0.0, sum(count / n * math.log2(count * n / (px[a] * py[b]))
                        for (a, b), count in joint.items()))


def roc_curve(scores, classes, positive=None):
    """Calcular una curva ROC sin bibliotecas de métricas.

    scores: iterable numérico; valores altos predicen la clase positiva.
    classes: etiquetas binarias de igual longitud que scores.
    positive: etiqueta positiva; None elige la última al ordenar como texto.
    Devuelve DataFrame con threshold, fpr, tpr, desde (0,0) con umbral Inf.
    Omite pares con ausentes; requiere ambas clases antes y después de omitirlos.
    Las puntuaciones empatadas se incorporan en un mismo punto.
    """
    scores, classes = _numeric(scores), list(classes)
    positive = _labels(classes, positive)
    pairs = sorted(_pairs(scores, classes), key=lambda pair: pair[0], reverse=True)
    positives = sum(label == positive for _, label in pairs)
    negatives = len(pairs) - positives
    if not positives or not negatives:
        raise ValueError("Se necesitan ambas clases entre los pares válidos.")
    rows = [(math.inf, 0.0, 0.0)]
    tp = fp = i = 0
    # Agrupar empates evita que el orden original altere la curva y su área.
    while i < len(pairs):
        threshold = pairs[i][0]
        while i < len(pairs) and pairs[i][0] == threshold:
            if pairs[i][1] == positive:
                tp += 1
            else:
                fp += 1
            i += 1
        rows.append((threshold, fp / negatives, tp / positives))
    return pd.DataFrame(rows, columns=["threshold", "fpr", "tpr"])


def auc(scores, classes, positive=None):
    """AUC en [0,1] mediante trapecios propios; entradas como roc_curve().

    Un empate aporta 1/2; puntuaciones constantes producen 0.5.
    """
    rows = list(roc_curve(scores, classes, positive).itertuples(index=False, name=None))
    return sum((b[1] - a[1]) * (a[2] + b[2]) / 2 for a, b in zip(rows, rows[1:]))


def attribute_metrics(df, class_column=None, positive=None, categorical=()):
    """Calcular métricas de todas las columnas excepto la clase de un DataFrame.

    class_column: nombre de clase binaria, o None (sin AUC).
    positive: etiqueta positiva para AUC.
    categorical: nombres que fuerzan tipo categórico, incluso códigos numéricos.
    Devuelve DataFrame con attribute, type, n, variance, entropy, auc.
    Columnas numéricas: varianza y AUC; otras, incluidas booleanas: entropía.
    No aplicables/indefinidas: NaN. Si no quedan ambas clases entre los pares de
    una columna, su AUC es NaN. Una clase global no binaria genera ValueError.
    """
    _frame(df, categorical)
    _class(df, class_column, positive)
    rows = []
    for name in df.columns:
        if name == class_column:
            continue
        column = df[name]
        numeric = _is_numeric(column) and name not in categorical
        v = h = a = math.nan
        if numeric:
            v = variance(column)
            if class_column is not None:
                pairs = _pairs(column, df[class_column])
                if len(set(label for _, label in pairs)) == 2:
                    a = auc(column, df[class_column], positive)
        else:
            h = entropy(column)
        rows.append((name, "numeric" if numeric else "categorical",
                     sum(not pd.isna(v) for v in column), v, h, a))
    return pd.DataFrame(rows, columns=["attribute", "type", "n", "variance", "entropy", "auc"])
