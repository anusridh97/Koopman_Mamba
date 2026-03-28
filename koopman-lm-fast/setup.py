from setuptools import setup, find_packages

setup(
    name="koopman-lm",
    version="0.2.0",
    packages=find_packages(),
    install_requires=[
        "torch>=2.2.0",
        "transformers>=4.48.0",
        "datasets>=2.18.0",
        "mamba-ssm>=2.0.0",
        "causal-conv1d>=1.2.0",
        "wandb",
        "safetensors",
        "numpy",
    ],
    python_requires=">=3.10",
)
