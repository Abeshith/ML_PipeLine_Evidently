from setuptools import setup, find_packages

setup(
    name="heartpipeline",
    version="0.0.1",
    author="Your Name",
    description="Heart Disease Prediction ML Pipeline with Evidently Monitoring",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "pandas",
        "numpy",
        "scikit-learn",
        "evidently",
        "pyyaml",
        "requests",
        "fastapi",
        "uvicorn",
    ],
    python_requires=">=3.8",
)
