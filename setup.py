from setuptools import setup, find_packages
from pathlib import Path

# Read requirements.txt safely
def read_requirements():
    req_path = Path(__file__).parent / "requirements.txt"
    if req_path.exists():
        return [
            line.strip()
            for line in req_path.read_text().splitlines()
            if line.strip() and not line.startswith("#")
        ]
    return []

setup(
    name="wave_propagation_2d",
    version="0.1.0",
    author="Carlos Celi",
    description="Python tools for 1D and 2D wave propagation using FDM and Pseudo-Spectral Method.",
    packages=find_packages(),
    install_requires=read_requirements(),
    python_requires=">=3.10",
)