from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="chiaroscuro_forge",
    version="0.1.0",
    author="Michail Semoglou",
    author_email="m.semoglou@tongji.edu.cn",
    description="An intelligent image enhancement tool inspired by Renaissance techniques",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/chiaroscuro-forge",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Multimedia :: Graphics",
        "Topic :: Scientific/Engineering :: Image Processing",
    ],
    python_requires=">=3.7",
    install_requires=[
        "numpy>=1.20.0",
        "scipy>=1.7.0",
        "scikit-image>=0.18.0",
    ],
    entry_points={
        "console_scripts": [
            "chiaroscuro-forge=chiaroscuro_forge:main",
        ],
    },
)
