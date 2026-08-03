from setuptools import setup, find_packages

import pathlib
import re

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()


def _read_version() -> str:
    # Read version from the package __init__.py
    init_path = pathlib.Path(__file__).parent / "chiaroscuro_forge" / "__init__.py"
    if not init_path.exists():
        raise RuntimeError("Unable to find chiaroscuro_forge/__init__.py")
    
    content = init_path.read_text(encoding="utf-8")
    match = re.search(r'^__version__\s*=\s*"([^"]+)"\s*$', content, re.M)
    if not match:
        raise RuntimeError("Unable to find __version__ in __init__.py")
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
    packages=find_packages(),
    package_data={"chiaroscuro_forge": ["presets/*.json"]},
    py_modules=["chiaroscuro_forge"] if not find_packages() else None,
    license="MIT",
    project_urls={
        "Source": "https://github.com/MichailSemoglou/chiaroscuro-forge",
        "Issues": "https://github.com/MichailSemoglou/chiaroscuro-forge/issues",
        "Documentation": "https://github.com/MichailSemoglou/chiaroscuro-forge/blob/main/README.md",
        "Changelog": "https://github.com/MichailSemoglou/chiaroscuro-forge/blob/main/CHANGELOG.md",
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Multimedia :: Graphics",
        "Topic :: Scientific/Engineering :: Image Processing",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.22.0",
        "scipy>=1.7.0",
        "scikit-image>=0.20.0",
        "Pillow>=10.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=24.4.1",
            "flake8>=6.1.0",
            "mypy>=1.5.0",
        ],
        "perceptual": [
            "lpips>=0.1.4",
            "torch>=2.0.0",
        ],
    },
    # Optional dependency extras:
    #   pip install chiaroscuro-forge[perceptual]
    #   Installs lpips and torch for perceptual metrics.
    entry_points={
        "console_scripts": [
            "chiaroscuro-forge=chiaroscuro_forge.cli:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)

