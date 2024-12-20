from setuptools import setup, find_packages

setup(
    name="GPcustom",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        # listez ici vos dépendances
        "numpy",
        "torch",
        "gpytorch",
        "botorch"
    ]
) 