# SME para Python

La distribución instalable se llama `sme-tools` y se importa como `sme`.
Se construye con setuptools, configurado en `setup.py`.

La entrega de Python contiene:

- `notebooks/entrega.ipynb`: notebook con ejemplos ejecutados de todas las funciones.
- `notebooks/entrega.html`: versión de lectura del mismo notebook.
- `src/*.py`: código fuente de la libreria.

El notebook usa únicamente Python y no necesita R ni acceso a internet para cargar datos.
Para AUC utiliza las observaciones de versicolor y virginica, con virginica como positiva.

Desde el directorio que contiene `SME-py`, instala el paquete y las herramientas del documento:

```bash
python3 -m pip install './SME-py[notebook]'
```

O instala únicamente el paquete:

```bash
python3 -m pip install ./SME-py
```


Los archivos se generan en `dist/`. La distribución fuente incluye el notebook,
su HTML y los datos. El wheel incluye el código del paquete y sus metadatos.

Ejemplo desde `SME-py` con el paquete instalado:

```python
import pandas as pd
from sme import Dataset

iris = pd.read_csv("data/iris.csv")
flowers = iris.loc[iris["Species"].isin(["versicolor", "virginica"])].copy()
ds = Dataset(flowers, class_column="Species", positive="virginica")
ds.metrics()
ds.normalize()
ds.discretize_ef(3)
ds.filter(min_auc=0.8)
ds.associations(num_bins=3)
ds.data.head()
```

`Dataset.to_csv()` y `Dataset.read_csv()` escriben y leen valores sin guardar los
metadatos del objeto: se especifican de nuevo al leer. Las funciones independientes
siguen disponibles mediante `import sme`.

`plot_pca()` añade una vista exploratoria de Iris completo: proyección PC1–PC2
coloreada por especie, varianza explicada individual y acumulada, y mapa de pesos
de los atributos. Centra y estandariza con desviación muestral por defecto;
`scale=False` permite conservar la escala original. El PCA usa la SVD de NumPy.
La especie solo colorea y se excluye del cálculo. Admite DataFrame y `Dataset`,
respetando la clase y las categorías declaradas del objeto.

```python
import matplotlib.pyplot as plt
from sme import plot_pca

pca = plot_pca(iris, group="Species", title="Iris · PCA estandarizado")
print(pca["explained_variance"])
pca["figure"].savefig("iris_pca.png", dpi=180, bbox_inches="tight")
plt.show()
```

La salida incluye `scores`, `loadings` (vectores unitarios), `center`, `scale`,
`dropped` (constantes omitidas), `figure` y `axes`. Requiere dos filas y dos
atributos numéricos no constantes. Rechaza ausentes e infinitos en los predictores
y ausentes en los grupos; las demás columnas no numéricas se excluyen.
