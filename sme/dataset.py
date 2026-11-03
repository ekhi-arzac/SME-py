"""Objeto para gestionar datos y metadatos con las funciones propias de SME."""

import pandas as pd

from ._utils import _class, _frame
from .association import association_matrix
from .discretization import discretize_ef, discretize_ew
from .metrics import attribute_metrics
from .preprocessing import normalize, standardize
from .selection import filter_features


class Dataset:
    """Dataset con clase binaria opcional y columnas categóricas declaradas.

    data: DataFrame de pandas con nombres únicos; se guarda una copia.
    class_column: nombre de clase binaria o None para datos no supervisados.
    positive: etiqueta positiva para AUC; None usa el criterio de roc_curve().
    categorical: nombres de columnas que representan categorías, incluidos códigos.

    Las transformaciones devuelven otro Dataset y protegen automáticamente la clase
    y las columnas categóricas declaradas. La propiedad data devuelve una copia.
    Las métricas reutilizan los metadatos sin tener que repetirlos en cada llamada.
    """

    def __init__(self, data, class_column=None, positive=None, categorical=()):
        categorical = tuple(dict.fromkeys(categorical))
        _frame(data, categorical)
        _class(data, class_column, positive)
        self._data = data.copy(deep=True)
        self._class_column = class_column
        self._positive = positive
        self._categorical = categorical

    @property
    def data(self):
        """Copia del DataFrame almacenado, con su índice y orden de columnas."""
        return self._data.copy(deep=True)

    @property
    def class_column(self):
        """Nombre de la clase binaria o None si no hay clase."""
        return self._class_column

    @property
    def positive(self):
        """Etiqueta positiva explícita o None para usar el criterio por defecto."""
        return self._positive

    @property
    def categorical(self):
        """Tupla con los nombres de las columnas declaradas categóricas."""
        return self._categorical

    def __repr__(self):
        """Resumen con dimensiones y columna de clase."""
        rows, columns = self._data.shape
        return f"Dataset({rows} filas, {columns} columnas, class_column={self.class_column!r})"

    @classmethod
    def read_csv(cls, path, class_column=None, positive=None, categorical=(), **kwargs):
        """Leer un CSV y devolver un Dataset; kwargs se pasan a pandas.read_csv.

        class_column, positive y categorical especifican de nuevo los metadatos:
        CSV solo guarda valores, no los metadatos ni los tipos originales. Para
        conservar códigos de texto puede indicarse dtype en kwargs.
        """
        return cls(pd.read_csv(path, **kwargs), class_column, positive, categorical)

    def to_csv(self, path, **kwargs):
        """Guardar valores sin índice en path; devuelve la ruta recibida.

        kwargs contiene opciones de DataFrame.to_csv, por ejemplo sep o na_rep.
        CSV no guarda los metadatos del objeto; al leer deben especificarse otra vez.
        """
        self._data.to_csv(path, index=False, **kwargs)
        return path

    def _replace(self, data, categorical=None):
        # Reconstruir valida los metadatos después de cada transformación.
        return type(self)(data, self.class_column, self.positive,
                          self.categorical if categorical is None else categorical)

    def _transform(self, fn, exclude):
        protected = list(dict.fromkeys((*self.categorical, *exclude)))
        if self.class_column is not None and self.class_column not in protected:
            protected.append(self.class_column)
        return self._replace(fn(self._data, exclude=protected))

    def normalize(self, exclude=()):
        """Nuevo Dataset normalizado; exclude permite proteger más columnas."""
        return self._transform(normalize, exclude)

    def standardize(self, exclude=()):
        """Nuevo Dataset estandarizado; exclude permite proteger más columnas."""
        return self._transform(standardize, exclude)

    def discretize_ew(self, num_bins, exclude=()):
        """Nuevo Dataset discretizado por igual anchura con num_bins intervalos.

        exclude añade columnas protegidas; se aplican los criterios de discretize_ew.
        """
        return self._transform(lambda data, exclude: discretize_ew(data, num_bins, exclude), exclude)

    def discretize_ef(self, num_bins, exclude=()):
        """Nuevo Dataset discretizado por igual frecuencia con num_bins intervalos.

        exclude añade columnas protegidas; se aplican los criterios de discretize_ef.
        """
        return self._transform(lambda data, exclude: discretize_ef(data, num_bins, exclude), exclude)

    def metrics(self):
        """Tabla de métricas con la clase, positiva y categorías del objeto."""
        return attribute_metrics(self._data, self.class_column, self.positive, self.categorical)

    def filter(self, min_variance=None, min_entropy=None, min_auc=None):
        """Nuevo Dataset con columnas que cumplen umbrales mínimos inclusivos.

        Reutiliza filter_features y los metadatos. Conserva la clase y retira de
        categorical los nombres de columnas eliminadas.
        """
        data = filter_features(self._data, min_variance, min_entropy, min_auc,
                               self.class_column, self.positive, self.categorical)
        categorical = tuple(name for name in self.categorical if name in data.columns)
        return self._replace(data, categorical)

    def associations(self, num_bins=5):
        """Dict values/methods de asociaciones usando los tipos del objeto.

        num_bins controla la discretización para pares mixtos. La clase, incluso
        si usa códigos numéricos, siempre se interpreta como categórica.
        """
        categorical = list(self.categorical)
        if self.class_column is not None and self.class_column not in categorical:
            categorical.append(self.class_column)
        return association_matrix(self._data, categorical, num_bins)
