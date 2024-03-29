from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='llm-rankers',
    version='0.0.1',
    packages=find_packages(),
    url='https://github.com/ielab/llm-rankers',
    license='Apache 2.0',
    author='Shengyao Zhuang',
    author_email='s.zhuang@uq.edu.au',
    description='Pointwise, Listwise, Pairwise and Setwise Document Ranking with Large Language Models.',
    python_requires='>=3.8',
    long_description=long_description,
    long_description_content_type="text/markdown",
    install_requires=[
        "transformers>=4.31.0",
        "openai>=0.27.10",
        "tiktoken>=0.4.0",
        "accelerate>=0.22.0"
    ]
)