from setuptools import find_packages, setup


setup(
    name="patchloop",
    version="0.1.0",
    description=(
        "Local-first GitHub agent loop for issue triage, PR review, "
        "patch generation, and review feedback repair"
    ),
    package_dir={"": "src"},
    packages=find_packages("src"),
    python_requires=">=3.10",
    install_requires=[
        "fastapi>=0.110.0",
        "uvicorn>=0.29.0",
        "httpx>=0.27.0",
        "python-dotenv>=1.0.0",
    ],
    entry_points={
        "console_scripts": [
            "patchloop=patchloop.cli:main",
        ],
    },
)
