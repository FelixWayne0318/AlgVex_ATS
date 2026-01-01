"""
AlgVex - Qlib + Hummingbot 融合的加密货币量化交易平台
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the requirements
requirements_path = Path("algvex") / "requirements.txt"
with open(requirements_path, encoding="utf-8") as f:
    requirements = [
        line.strip()
        for line in f
        if line.strip() and not line.startswith("#") and not line.startswith("-")
    ]

# Read README
readme_path = Path("algvex") / "README.md"
long_description = ""
if readme_path.exists():
    with open(readme_path, encoding="utf-8") as f:
        long_description = f.read()

setup(
    name="algvex",
    version="2.0.0",
    description="AlgVex - Qlib + Hummingbot 融合的加密货币量化交易平台",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="AlgVex Team",
    packages=find_packages(where=".", include=["algvex", "algvex.*"]),
    python_requires=">=3.10",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.3.0",
            "pytest-asyncio>=0.21.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "flake8>=6.0.0",
            "mypy>=1.2.0",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Financial and Insurance Industry",
        "Topic :: Office/Business :: Financial :: Investment",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)
