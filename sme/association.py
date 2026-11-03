"""Asociaciones propias por pares de variables."""

import math

import pandas as pd

from ._utils import _bins, _frame, _is_numeric, _numeric, _pairs
from .discretization import discretize_ef
from .metrics import mean, mutual_info


def pearson(x, y):
    """Pearson de dos iterables numéricos de igual longitud; omite pares ausentes.

    Devuelve coeficiente en [-1,1], o NaN con constantes o menos de dos pares.
    Calcula sum(dx*dy)/sqrt(sum(dx²)*sum(dy²)) sin funciones de correlación.
    """
    pairs = _pairs(_numeric(x), _numeric(y))
    if len(pairs) < 2:
        return math.nan
    x, y = zip(*pairs)
    mx, my = mean(x), mean(y)
    dx, dy = [v - mx for v in x], [v - my for v in y]
    denominator = math.sqrt(sum(v * v for v in dx)) * math.sqrt(sum(v * v for v in dy))
    if not denominator:
        return math.nan
    return max(-1.0, min(1.0, sum(a * b for a, b in zip(dx, dy)) / denominator))


def association_matrix(df, categorical=(), num_bins=5):
    """Calcular asociaciones de todos los pares de columnas de un DataFrame.

    categorical: nombres que fuerzan tipo categórico; num_bins: entero positivo.
    Dos numéricas usan Pearson; dos categóricas, información mutua en bits.
    Pares mixtos: discretiza la numérica por igual frecuencia sobre pares válidos
    y calcula información mutua. Elimina pares ausentes antes del cálculo.
    Devuelve dict con DataFrames simétricos 'values' y 'methods' (pearson, mi,
    mi_discretized). La diagonal categórica es la entropía, no necesariamente 1.
    """
    _frame(df, categorical)
    _bins(num_bins)
    names = list(df.columns)
    values = pd.DataFrame(math.nan, index=names, columns=names)
    methods = pd.DataFrame("", index=names, columns=names)
    numeric = [_is_numeric(df[name]) and name not in categorical for name in names]
    for i, name in enumerate(names):
        for j in range(i + 1):
            other = names[j]
            x, y = df[name], df[other]
            if numeric[i]:
                x = _numeric(x)
            if numeric[j]:
                y = _numeric(y)
            pairs = _pairs(x, y)
            x, y = zip(*pairs) if pairs else ([], [])
            if numeric[i] and numeric[j]:
                value, method = pearson(x, y), "pearson"
            else:
                if numeric[i]:
                    x = discretize_ef(x, num_bins)
                if numeric[j]:
                    y = discretize_ef(y, num_bins)
                value = mutual_info(x, y)
                method = "mi_discretized" if numeric[i] or numeric[j] else "mi"
            values.loc[name, other] = values.loc[other, name] = value
            methods.loc[name, other] = methods.loc[other, name] = method
    return {"values": values, "methods": methods}
