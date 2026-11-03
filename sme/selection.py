"""Filtrado de atributos mediante las métricas propias."""

import math
from numbers import Real

from .metrics import attribute_metrics


def filter_features(df, min_variance: float =None, min_entropy=None, min_auc=None,
                    class_column=None, positive=None, categorical=()):
    """Seleccionar columnas que cumplen todos sus umbrales mínimos inclusivos.

    df: DataFrame. min_variance, min_entropy, min_auc: escalares finitos o None.
    Varianza/AUC se aplican a numéricas; entropía, a categóricas.
    class_column: clase binaria, obligatoria si min_auc no es None; se conserva.
    positive: etiqueta positiva. categorical: columnas que fuerzan tipo categórico.
    Devuelve copia del DataFrame con columnas seleccionadas en su orden original.
    Una métrica indefinida no supera su umbral; sin umbral se conserva la columna.
    """
    thresholds = {"variance": min_variance, "entropy": min_entropy, "auc": min_auc}
    for threshold in thresholds.values():
        if threshold is not None and (isinstance(threshold, bool) or
                not isinstance(threshold, Real) or not math.isfinite(threshold)):
            raise ValueError("Los umbrales deben ser números finitos.")
    if min_auc is not None and class_column is None:
        raise ValueError("El filtrado por AUC requiere clase.")
    metrics = attribute_metrics(df, class_column, positive, categorical)
    keep = {class_column} if class_column is not None else set()
    for row in metrics.to_dict("records"):
        accepted = True
        for metric, threshold in thresholds.items():
            applicable = row["type"] == ("categorical" if metric == "entropy" else "numeric")
            if threshold is not None and applicable:
                accepted = accepted and not math.isnan(row[metric]) and row[metric] >= threshold
        if accepted:
            keep.add(row["attribute"])
    return df.loc[:, [name for name in df.columns if name in keep]].copy()
