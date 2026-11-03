"""Validaciones y selección de datos compartidas (uso interno)."""

import math
from numbers import Integral, Real

import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype


def _numeric(values):
    result = []
    for value in values:
        if pd.isna(value):
            result.append(None)
        elif isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError("Se requiere un vector numérico real.")
        elif not math.isfinite(value):
            raise ValueError("No se admiten infinitos.")
        else:
            result.append(float(value))
    return result


def _bins(n):
    if isinstance(n, bool) or not isinstance(n, Integral) or n < 1:
        raise ValueError("num_bins debe ser un entero positivo.")


def _frame(df, columns=()):
    if not isinstance(df, pd.DataFrame) or not df.columns.is_unique:
        raise TypeError("Se requiere un DataFrame con nombres únicos.")
    if any(name not in df.columns for name in columns):
        raise ValueError("Hay nombres de columnas desconocidos.")


def _is_numeric(column):
    return is_numeric_dtype(column.dtype) and not is_bool_dtype(column.dtype)


def _transform(data, fn, exclude):
    if isinstance(data, pd.DataFrame):
        _frame(data, exclude)
        result = data.copy()
        for name in data.columns:
            if name not in exclude and _is_numeric(data[name]):
                result[name] = fn(data[name])
        return result
    # Las Series conservan su índice; los demás iterables devuelven listas.
    result = fn(data)
    if isinstance(data, pd.Series):
        return pd.Series(result, index=data.index, name=data.name)
    return result


def _pairs(x, y):
    x, y = list(x), list(y)
    if len(x) != len(y):
        raise ValueError("Los vectores deben tener la misma longitud.")
    return [(a, b) for a, b in zip(x, y) if not pd.isna(a) and not pd.isna(b)]


def _labels(values, positive):
    # El orden por representación de texto coincide con el criterio de R.
    labels = sorted(set(v for v in values if not pd.isna(v)), key=str)
    if len(labels) != 2:
        raise ValueError("La clase debe ser binaria.")
    if positive is None:
        positive = labels[-1]
    if pd.isna(positive) or positive not in labels:
        raise ValueError("La etiqueta positiva no pertenece a la clase.")
    return positive


def _class(df, class_column, positive):
    _frame(df)
    if class_column is None:
        if positive is not None:
            raise ValueError("positive requiere una columna de clase.")
    else:
        _frame(df, [class_column])
        _labels(df[class_column], positive)
