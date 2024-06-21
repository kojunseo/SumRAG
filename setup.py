from setuptools import setup, find_packages

with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(
    name='SumRAG',
    version='0.0.1',
    author="Junseo Ko",
    description="RAG with summarization and LLM based retrieve for chapter-based documents",
    packages=find_packages(include=['SumRAG', 'SumRAG.*'),
    install_requires=required,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ]
)
