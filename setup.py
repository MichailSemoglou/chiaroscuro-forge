from setuptools import setup

import pathlib
import re

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()


def _read_version() -> str:
    module_path = pathlib.Path(__file__).parent / "chiaroscuro_forge.py"
    content = module_path.read_text(encoding="utf-8")
    match = re.search(r"^__version__\s*=\s*\"([^\"]+)\"\s*$", content, re.M)
    if not match:
        raise RuntimeError("Unable to find __version__ in chiaroscuro_forge.py")
    return match.group(1)

setup(
    # Distribution name on PyPI (can differ from import name).
    name="chiaroscuro-forge",
    version=_read_version(),
    author="Michail Semoglou",
    author_email="m.semoglou@tongji.edu.cn",
    description="An intelligent image enhancement tool inspired by Renaissance techniques",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/MichailSemoglou/chiaroscuro-forge",
    py_modules=["chiaroscuro_forge"],
    license="MIT",
    project_urls={
        "Source": "https://github.com/MichailSemoglou/chiaroscuro-forge",
        "Issues": "https://github.com/MichailSemoglou/chiaroscuro-forge/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3 :: Only",
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
