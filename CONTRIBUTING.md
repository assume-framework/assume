<!--
SPDX-FileCopyrightText: ASSUME Developers

SPDX-License-Identifier: AGPL-3.0-or-later
-->

# Contribute to ASSUME

Everyone is welcome to contribute, and we value everybody's contribution. Code
contributions are not the only way to help the community. Answering questions, helping
others, and improving the documentation are also immensely valuable.

It also helps us if you spread the word! Reference the framework in blog posts
about the awesome projects it made possible, use it in your research papers or thesis, or simply ⭐️ the repository to say thank you.

However you choose to contribute, please be mindful and respect our
[code of conduct](./CODE_OF_CONDUCT.md).

The user documentation is also a valuable source for developers and can be found here:

- [User Documentation](https://assume.readthedocs.io/en/latest/)
- [Installation Guide](https://assume.readthedocs.io/en/latest/installation.html)

## Bug reports and issues

You can file issues in our Issue tracker at: https://github.com/assume-framework/assume/issues

For this, a GitHub account is required.

## Development setup

If you're contributing to the development of ASSUME, follow these steps:
1. Clone the repository and navigate to its directory:

```bash
git clone https://github.com/assume-framework/assume.git
cd assume
```

2. Install the package in editable mode:

```bash
pip install -e ".[all]"
```

3. Install pre-commit and configure it to run prior to every commit

```bash
pip install pre-commit
pre-commit install
```

To run pre-commit checks directly, use:

```bash
pre-commit run --all-files
```

4. Install also testing capabilities:

```bash
pip install -e ".[testing]"
```

5. Implement your changes and push the changes to a fork

6. Create a pull request at https://github.com/assume-framework/assume/pulls

### Creating a new release

To release a new version, create a git tag of the release commit and release notes in GitHub (also possible via `Draft new release` function of Github). No adjusted toml file necessary anymore.
A github action automatically uploads the release to PyPI.
The upload to PyPi has to be confirmed by one of the core developers.

## Building documentation

First, create an environment that includes the documentation dependencies:


### Conda

Conda installs the required dependencies and pandoc as well:

```bash
conda env create -f environment_docs.yaml
```

### Venv

To use a venv without conda, a system installation of pandoc is required: https://pandoc.org/installing.html

You then need to install the dependencies into your venv using:

```bash
pip install -e .[docs]
```

### Building Docs

To generate or update the automatically created docs in `docs/source/assume*`, run:

```bash
sphinx-apidoc -o docs/source -Fa assume
```

To create and serve the documentation locally, use:

```bash
cd docs/source && python -m sphinx . ../build && cd ../.. && python -m http.server --directory docs/build
```

## Need some help?

Reach out to us with your questions:

kim.miskiw@kit.edu / manish.khanra@isi.fraunhofer.de / maurer@fh-aachen.de / nick.harder@inatech.uni-freiburg.de
