# Contributing to Chiaroscuro Forge

Thanks for considering contributing to Chiaroscuro Forge.

## Development Process

### Pull Requests

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Pull Request Process

1. Ensure any install or build dependencies are removed before the end of the layer when doing a build.
2. Update the README.md with details of changes to the interface, if appropriate.
3. The versioning scheme we use is [SemVer](http://semver.org/).
4. You may merge the Pull Request once you have the sign-off of another developer, or if you do not have permission to do that, you may request the reviewer to merge it for you.

## Code Style

We use [Black](https://github.com/psf/black) with a line length of 100, [isort](https://github.com/PyCQA/isort) with the `black` profile, and [flake8](https://flake8.pycqa.org/) for linting. All code must pass the following checks before submission:

```bash
black --check --line-length 100 chiaroscuro_forge/ tests/
isort --check-only --profile black chiaroscuro_forge/ tests/
flake8 chiaroscuro_forge/
```

Docstrings follow the [numpydoc](https://numpydoc.readthedocs.io/) convention. The docstring and type-checking commands below are local-only checks and are not part of the CI lint job:

```bash
pip install flake8-docstrings mypy
flake8 --extend-select=D chiaroscuro_forge/
mypy chiaroscuro_forge/
```

## Testing

All tests must pass and new features must include tests. Coverage should not decrease below the existing bar.

```bash
# Run the full suite
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=chiaroscuro_forge --cov-report=term --cov-report=html

# Run only the quick subset (used locally during development)
pytest tests/ -q --no-cov --ignore=tests/test_property_based.py --ignore=tests/test_gpu.py -k "not test_tiling"
```

## Lint, format, and coverage checklist

```bash
# Format
black --line-length 100 chiaroscuro_forge/ tests/
isort --profile black chiaroscuro_forge/ tests/

# Lint
flake8 chiaroscuro_forge/

# Type check
mypy chiaroscuro_forge/

# Test with coverage
pytest tests/ --cov=chiaroscuro_forge --cov-fail-under=64
```

## Issue Reporting Guidelines

When reporting issues, please include:

- Your operating system name and version
- Python version
- Detailed steps to reproduce the bug
- What you expected to happen
- What actually happened
- Sample images (if applicable and shareable)

## Feature Requests

We love to hear your ideas for new features. Please use the GitHub issue tracker to submit feature requests.

## Community

Discussions about Chiaroscuro Forge take place on this repository's Issues and Pull Requests sections. Anybody is welcome to join these conversations.

## License

By contributing, you agree that your contributions will be licensed under the project's MIT License.
