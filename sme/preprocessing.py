"""Transformaciones numéricas para vectores y DataFrames."""

import math

from ._utils import _numeric, _transform
from .metrics import mean, std_dev


def normalize(x, exclude=()):
    """Normalizar por min-max a [0,1] un vector, Series o DataFrame.

    exclude: nombres de columnas que se conservan (p. ej., la clase).
    Devuelve lista, Series con el mismo índice, o copia del DataFrame.
    Conserva ausentes y columnas no numéricas; constantes se convierten en ceros.
    """
    def transform(column):
        values = _numeric(column)
        observed = [v for v in values if v is not None]
        if not observed:
            return values
        lower, upper = min(observed), max(observed)
        return [None if v is None else (v - lower) / (upper - lower) if upper != lower else 0.0
                for v in values]

    return _transform(x, transform, exclude)


def standardize(x, exclude=()):
    """Estandarizar usando media y desviación típica muestral propias.

    x: vector, Series o DataFrame. exclude: columnas que se conservan.
    Devuelve lista, Series con el mismo índice, o copia del DataFrame.
    Solo transforma numéricas, conserva ausentes; constantes o una sola
    observación generan ceros. Las demás tienen media 0 y desviación muestral 1.
    """
    def transform(column):
        values = _numeric(column)
        center, deviation = mean(values), std_dev(values)
        return [None if v is None else 0.0 if math.isnan(deviation) or deviation == 0
                else (v - center) / deviation for v in values]

    return _transform(x, transform, exclude)
