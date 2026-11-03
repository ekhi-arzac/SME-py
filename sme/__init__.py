"""Funciones propias de SME: solo pandas para datos y matplotlib para gráficos."""

from .association import association_matrix, pearson
from .dataset import Dataset
from .discretization import discretize, discretize_ef, discretize_ew, find_range
from .metrics import (attribute_metrics, auc, cross_entropy, entropy, mean,
                      mutual_info, roc_curve, std_dev, variance)
from .plots import plot_association, plot_roc
from .preprocessing import normalize, standardize
from .selection import filter_features

__all__ = [
    "Dataset",
    "association_matrix", "pearson", "discretize", "discretize_ef", "discretize_ew",
    "find_range", "attribute_metrics", "auc", "cross_entropy", "entropy", "mean",
    "mutual_info", "roc_curve", "std_dev", "variance", "plot_association", "plot_roc",
    "normalize", "standardize", "filter_features",
]
