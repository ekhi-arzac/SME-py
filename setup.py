"""Configuración de setuptools para construir e instalar SME para Python."""

from pathlib import Path

from setuptools import find_packages, setup


ROOT = Path(__file__).resolve().parent

setup(
    name="sme-tools",
    version="0.1.0",
    description="Implementaciones propias para preparar y analizar atributos de datasets",
    long_description=(ROOT / "README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    author="Ekhi Arzac",
    project_urls={"Repository": "https://github.com/ekhi-arzac/SME-py"},
    packages=find_packages(include=["sme", "sme.*"]),
    python_requires=">=3.10",
    install_requires=["pandas>=2.0", "matplotlib>=3.7", "numpy>=1.24"],
    extras_require={"notebook": ["jupyter>=1.0", "nbconvert>=7.0"]},
)
