<!--
SPDX-FileCopyrightText: ASSUME Developers

SPDX-License-Identifier: AGPL-3.0-or-later
-->

# ASSUME: Agent-Based Electricity Markets Simulation Toolbox

![Lint Status](https://github.com/assume-framework/assume/actions/workflows/lint-pytest.yaml/badge.svg)
[![Code Coverage](https://codecov.io/gh/assume-framework/assume/branch/main/graph/badge.svg?token=CZ4FO7P57H)](https://codecov.io/gh/assume-framework/assume)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.8088760.svg)](https://doi.org/10.5281/zenodo.8088760)
[![REUSE status](https://api.reuse.software/badge/github.com/assume-framework/assume)](https://api.reuse.software/info/github.com/assume-framework/assume)

[![](https://img.shields.io/pypi/v/assume-framework.svg)](https://pypi.org/pypi/assume-framework/)
[![](https://img.shields.io/pypi/pyversions/assume-framework.svg)](https://pypi.org/pypi/assume-framework/)
[![](https://img.shields.io/pypi/l/assume-framework.svg)](https://pypi.org/pypi/assume-framework/)
[![](https://img.shields.io/pypi/status/assume-framework.svg)](https://pypi.org/pypi/assume-framework/)
[![](https://img.shields.io/readthedocs/assume)](https://assume.readthedocs.io/)

[![Try examples in Collab](https://colab.research.google.com/assets/colab-badge.svg)](https://github.com/assume-framework/assume/tree/main/examples/notebooks)

**ASSUME** is an open-source toolbox for agent-based simulations of European electricity markets, with a primary focus on the German market setup. Developed as an open-source model, its primary objectives are to ensure usability and customizability for a wide range of users and use cases in the energy system modeling community.

## Introduction

A unique feature of the ASSUME toolbox is its integration of **Deep Reinforcement Learning** methods into the behavioral strategies of market agents. The model offers various predefined agent representations for both the demand and generation sides, which can be used as plug-and-play modules, simplifying the reinforcement of learning strategies. This setup enables research into new market designs and dynamics in energy markets.


## Documentation

- [User Documentation](https://assume.readthedocs.io/en/latest/)
- [Installation Guide](https://assume.readthedocs.io/en/latest/installation.html)

## Installation

You can install ASSUME using pip. Choose the appropriate installation method based on your needs:

### Using pip

To install the core package:

```bash
pip install assume-framework
```

To install with reinforcement learning capabilities:

```bash
pip install assume-framework[learning]
```

We also include market clearing algorithms for complex bids. These clearing algorithms require pyomo and a solver (we use GLPK). To install the package with these capabilities, use:

```bash
pip install assume-framework[optimization]
```

To install with testing capabilities:

```bash
pip install assume-framework[test]
```

### Timescale Database and Grafana Dashboards

If you want to benefit from a supported database and integrated Grafana dashboards for scenario analysis, you can use the provided Docker Compose file.

Follow these steps:

1. Clone the repository and navigate to its directory:

```bash
git clone https://github.com/assume-framework/assume.git
cd assume
```

2. Start the database and Grafana using the following command:

```bash
docker-compose up -d
```

This will launch a container for TimescaleDB and Grafana with preconfigured dashboards for analysis. You can access the Grafana dashboards at `http://localhost:3000`.

### Using Learning Capabilities

If you intend to use the reinforcement learning capabilities of ASSUME and train your agents, make sure to install Torch. Detailed installation instructions can be found [here](https://pytorch.org/get-started/locally/).



## Trying out ASSUME and the provided Examples

To ease your way into ASSUME we provided some examples and tutorials. The former are helpful if you would like to get an impression of how ASSUME works and the latter introduce you into the development of ASSUME.

### The Tutorials

The tutorials work completly detached from your own machine on google colab. They provide code snippets and task that show you, how you can work with the software package one your own. We have two tutorials prepared, one for introducing a new unit and one for getting reinforcement learning ready on ASSUME.

How to configure a new unit in ASSUME?
**Coming Soon**

How to introduce reinforcement learning to ASSUME?

[![Open Learning Tutorial in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/assume-framework/assume/blob/main/examples/notebooks/04_reinforcement_learning_example.ipynb)



### The Examples

To explore the provided examples, follow these steps:

1. Clone the repository and navigate to its directory:

```bash
git clone https://github.com/assume-framework/assume.git
cd assume
```

2. Quick Start:

There are three ways to run a simulation:

- Local:

```bash
python examples/examples.py
```

- Using the provided Docker setup:

If you have installed Docker and set up the Docker Compose file previously, you can select 'timescale' in `examples.py` before running the simulation. This will save the simulation results in a Timescale database, and you can access the Dashboard at `http://localhost:3000`.

- Using the CLI to run simulations:

```bash
assume -s example_01b -db "postgresql://assume:assume@localhost:5432/assume"
```

For additional CLI options, run `assume -h`.

## Development

If you're contributing to the development of ASSUME, follow these steps:

1. Install pre-commit:

```bash
pip install pre-commit
pre-commit install
```

To run pre-commit checks directly, use:

```bash
pre-commit run --all-files
```

## Creating Documentation

First, create an environment that includes the documentation dependencies:

```bash
conda env create -f environment_docs.yaml
```

To generate or update the automatically created docs in `docs/source/assume*`, run:

```bash
sphinx-apidoc -o docs/source -Fa assume
```

To create and serve the documentation locally, use:

```bash
make -C docs html && python -m http.server --directory docs/build/html
```

## Contributors and Funding

The project is developed by a collaborative team of researchers from INATECH at the University of Freiburg, IISM at Karlsruhe Institute of Technology, Fraunhofer Institute for Systems and Innovation Research, Fraunhofer Institution for Energy Infrastructures and Geothermal Energy, and FH Aachen - University of Applied Sciences. Each contributor brings valuable expertise in electricity market modeling, deep reinforcement learning, demand side flexibility, and infrastructure modeling.

ASSUME is funded by the Federal Ministry for Economic Affairs and Climate Action (BMWK). We are grateful for their support in making this project possible.

## License

Copyright 2022-2024 [ASSUME developers](https://assume.readthedocs.io/en/latest/developers.html).

ASSUME is licensed under the [GNU Affero General Public License v3.0](./LICENSES/AGPL-3.0-or-later.txt). This license is a strong copyleft license that requires that any derivative work be licensed under the same terms as the original work. It is approved by the [Open Source Initiative](https://opensource.org/licenses/AGPL-3.0).
