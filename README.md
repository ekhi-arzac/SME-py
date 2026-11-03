# SME para Python

La distribución instalable se llama `sme-tools` y se importa como `sme`.
Se construye con setuptools, configurado en `setup.py`.

La entrega de Python utiliza Iris y contiene:

- `notebooks/entrega.ipynb`: notebook con ejemplos ejecutados de todas las funciones.
- `notebooks/entrega.html`: versión de lectura del mismo notebook.
- `data/iris.csv`: datos originales de Iris, con procedencia en `data/README.md`.

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

Para volver a ejecutar y exportar el notebook, desde `SME-py`:

```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/entrega.ipynb
jupyter nbconvert --to html --embed-images notebooks/entrega.ipynb
```

Para generar wheel y distribución fuente, también desde `SME-py`:

```bash
python3 -m pip install setuptools wheel
python3 setup.py sdist bdist_wheel
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
