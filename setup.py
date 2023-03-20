#!/usr/bin/env python
# coding=utf-8
"""The setup script."""
import ast

from setuptools import find_packages, setup  # type: ignore

with open("ahcore/__init__.py") as f:
    for line in f:
        if line.startswith("__version__"):
            version = ast.parse(line).body[0].value.s  # type: ignore
            break


with open("README.md") as readme_file:
    long_description = readme_file.read()

install_requires = [
    "numpy>=1.23.2",
    "torch>=1.12",
    "pillow>=9.4.0",
    "pytorch-lightning>=1.9.1",
    "torchvision>=0.13.1",
    "pydantic>=1.10.4",
    "tensorboard>=2.9",
    "mlflow>=1.26",
    "hydra-core>=1.2.0",
    "python-dotenv>=0.20",
    "tqdm>=4.64",
    "rich>=12.4",
    "hydra-submitit-launcher>=1.2.0",
    "hydra-optuna-sweeper>=1.2.0",
    "hydra-colorlog>=1.2.0",
    "dlup>=0.3.24",
    "kornia>=0.6.9",
    "h5py",
    "escnn",  # To use equivariant unets
]


setup(
    author="AI for Oncology Lab @ The Netherlands Cancer Institute",
    long_description=long_description,
    long_description_content_type="text/markdown",
    python_requires=">=3.9",
    classifiers=[
        "Development Status :: 1 - Planning",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
    ],
    entry_points={
        "console_scripts": [
            "ahcore=ahcore.cli:main",
        ],
    },
    description="Ahcore the AI for Oncology core components for computational pathology.",
    install_requires=install_requires,
    extras_require={
        "dev": [
            "pytest",
            "numpydoc",
            "pylint",
            "black==22.6.0",
            "types-Pillow",
            "sphinx",
            "sphinx_copybutton",
            "numpydoc",
            "myst-parser",
            "sphinx-book-theme",
        ],
    },
    license="Apache Software License 2.0",
    include_package_data=True,
    name="ahcore",
    test_suite="tests",
    url="https://github.com/NKI-AI/ahcore",
    py_modules=["ahcore"],
    version=version,
)
