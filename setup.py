from setuptools import find_packages, setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()


setup(
    name="russian_paraphrasers",
    version="0.0.3",
    author="Alenusch",
    author_email="alenush93@gmail.com",
    description="Russian Paraphrasers (based on ru-gpt, mt5)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/RussianNLP/russian_paraphrasers",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=["nltk", "scipy", "transformers>=4.0.1",
                      "sentence-transformers==0.4.0",
                      "nlg-eval @ git+https://github.com/Maluuba/nlg-eval.git@master"],
    setup_requires=[]
)