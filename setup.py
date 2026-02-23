from setuptools import setup, find_packages

setup(
    name="qhi-probe",
    version="0.1.0",
    author="Pranav",
    description="QHI-Probe: Quantified Hallucination Index for Clinical LLMs",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/pranav-qhi-probe/qhi-probe",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21.0",
        "scikit-learn>=1.0.0",
        "pandas>=1.3.0",
    ],
    extras_require={
        "datasets": ["datasets>=2.0.0"],
        "transformers": ["transformers>=4.30.0", "torch>=2.0.0"],
        "dev": ["pytest>=7.0", "black", "isort"],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Healthcare Industry",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
    ],
    keywords=[
        "hallucination", "clinical NLP", "LLM safety",
        "medical AI", "probing", "interpretability"
    ],
)
