"""Discretización propia para vectores y DataFrames; no usa cut ni qcut."""

import pandas as pd

from ._utils import _bins, _numeric, _transform


def find_range(value, cut_points):
    """Devolver índice desde 1 para un número y cortes ordenados sin ausentes.

    Un valor igual a un corte pertenece al intervalo inferior. Un valor
    ausente devuelve None. Utiliza búsqueda binaria sobre los cortes.
    """
    if pd.isna(value):
        return None
    left, right = 0, len(cut_points)
    while left < right:
        middle = (left + right) // 2
        if value <= cut_points[middle]:
            right = middle
        else:
            left = middle + 1
    return left + 1


def discretize(x, cut_points, label_prefix="I"):
    """Etiquetar un vector numérico con cortes explícitos y devolver una lista.

    x: iterable numérico; admite ausentes, rechaza infinitos.
    cut_points: iterable de cortes finitos sin ausentes; se ordena y deduplica.
    label_prefix: cadena usada delante del índice, por defecto 'I'.
    Salida: lista de etiquetas I1, I2, ... y None para ausentes.
    """
    values = _numeric(x)
    cuts = _numeric(cut_points)
    if any(v is None for v in cuts):
        raise ValueError("Los cortes no pueden contener ausentes.")
    if not isinstance(label_prefix, str):
        raise TypeError("label_prefix debe ser una cadena.")
    cuts = sorted(set(cuts))
    return [None if v is None else f"{label_prefix}{find_range(v, cuts)}" for v in values]


def discretize_ew(x, num_bins, exclude=()):
    """Discretizar por igual anchura un vector, Series o DataFrame.

    num_bins: entero positivo. exclude: nombres de columnas que se conservan.
    Devuelve lista, Series (con su índice) o copia del DataFrame, respectivamente.
    Solo transforma columnas numéricas; booleanas y otras categorías se conservan.
    Constantes generan I1; conserva ausentes. Los límites van al grupo inferior.
    """
    _bins(num_bins)

    def transform(column):
        values = _numeric(column)
        observed = [v for v in values if v is not None]
        cuts = []
        if observed and min(observed) != max(observed):
            lower, upper = min(observed), max(observed)
            cuts = [lower + (upper - lower) * i / num_bins for i in range(1, num_bins)]
        return discretize(values, cuts)

    return _transform(x, transform, exclude)


def discretize_ef(x, num_bins, exclude=()):
    """Discretizar por igual frecuencia sin separar valores iguales.

    x: vector, Series o DataFrame. num_bins: entero positivo.
    exclude: nombres de columnas que no se transforman en un DataFrame.
    Devuelve lista, Series o copia del DataFrame; conserva ausentes y no numéricas.
    Los cortes son valores en posiciones floor(n*j/num_bins), desde 1; se eliminan
    cortes duplicados y el máximo. Los empates pueden reducir el número de grupos
    y producir frecuencias desiguales. Una constante genera I1.
    """
    _bins(num_bins)

    def transform(column):
        values = _numeric(column)
        ordered = sorted(v for v in values if v is not None)
        cuts = []
        if len(ordered) >= 2:
            positions = [max(1, len(ordered) * j // num_bins) for j in range(1, num_bins)]
            cuts = sorted(set(ordered[p - 1] for p in positions if ordered[p - 1] < ordered[-1]))
        return discretize(values, cuts)

    return _transform(x, transform, exclude)
