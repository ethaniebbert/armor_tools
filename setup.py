from setuptools import setup, find_packages
from pathlib import Path

# Read requirements from requirements.txt
requirements_path = Path(__file__).parent / "requirements.txt"
with open(requirements_path) as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="armor_tools",
    version="0.1",
    packages=find_packages(),
    install_requires=requirements,
    python_requires=">=3.10",
)
