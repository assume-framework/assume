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
[![DOI](https://joss.theoj.org/papers/a8843ad1978808dc593b16437a2a029e/status.svg)](https://joss.theoj.org/papers/a8843ad1978808dc593b16437a2a029e)

[![Try examples in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://github.com/assume-framework/assume/tree/main/examples/notebooks)

**ASSUME** is an open-source toolbox for agent-based simulations of European electricity markets, with a primary focus on the German market setup.
Developed as an open-source model, its primary objectives are to ensure usability and customizability for a wide range of users and use cases in the energy system modeling community.

## Introduction

A unique feature of the ASSUME toolbox is its integration of **Deep Reinforcement Learning** methods into the behavioral strategies of market agents. The model offers various predefined agent representations for both the demand and generation sides, which can be used as plug-and-play modules, simplifying the reinforcement of learning strategies.
This setup enables research into new market designs and dynamics in energy markets.

### What can the software do?
The main motivation of ASSUME is to overcome the limitations of fixed, rule-based behaviors in existing ABMs.
For this, it leverages advancements in artificial intelligence, particularly deep reinforcement learning (DRL), enabling agents to adapt their behavior dynamically in response to market conditions.
This approach is practical for modeling interactions among competing market participants and observing emergent patterns from these interactions.
To support market design analysis in transforming electricity systems, we developed the ASSUME framework - a flexible and modular agent-based modeling tool for electricity market research.
ASSUME enables researchers to customize components such as agent representations, market configurations, and bidding strategies, utilizing pre-built modules for standard operations.
With the setup in ASSUME, researchers can simulate strategic interactions in electricity markets under a wide range of scenarios, from comparing market designs and modeling congestion management to analyzing the behavior of learning storage operators and renewable producers.
The framework supports studies on bidding under uncertainty, regulatory interventions, and multi-agent dynamics, making it ideal for exploring emergent behavior and testing new market mechanisms.
ASSUME has been utilized in research studies addressing diverse questions in electricity market design and operation.
It has explored the role of complex bids, demonstrated the effects of industrial demand-side flexibility for congestion management, and advanced the explainability of emergent strategies in learning agents.


### Who is it made for?
The framework is versatile enough to be employed in smaller-scale projects, such as master's theses, while also robust enough for complex doctoral research or investigations conducted by industry professionals.
This accessibility and scalability make ASSUME suitable for many users, from early-career researchers to experienced professionals.


## Documentation

- [User Documentation](https://assume.readthedocs.io/en/latest/)
- [Installation Guide](https://assume.readthedocs.io/en/latest/installation.html)

## Installation

You can install ASSUME using [pip](https://pip.pypa.io/).
Choose the appropriate installation method based on your needs:

### Using pip

To install the core package:

```bash
pip install assume-framework
```

**To install with reinforcement learning capabilities:**

```bash
pip install 'assume-framework[learning]'
```

Please keep in mind, that the above installation method will install pytorch package without CUDA support.
If you want to make use of your GPU with CUDA cores, please install pytorch with GPU support separately as described [here](https://pytorch.org/get-started/locally/).

We also include **network-based market clearing algorithms** such as for the re-dispatch, zonal clearing with NTCs and nodal market clearing, which all require the PyPSA library.
To install the package with these capabilities, use:

```bash
pip install 'assume-framework[network]'
```

To install with all capabilities:

```bash
pip install 'assume-framework[all]'
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
docker compose up -d
```

This will launch a container for TimescaleDB and Grafana with preconfigured dashboards for analysis.
You can access the Grafana dashboards at `http://localhost:3000`.

### Using TensorBoard to display Learning Metrics

When running an example with learning capabilities, you can start TensorBoard to observe the learning process.
Use the following shell command to start TensorBoard:
```shell
tensorboard --logdir tensorboard
```

You can then head to `http://localhost:6006/` to view and evaluate the training progress.

Please note that TensorBoard should ideally be shut down via `Ctrl + C` every time you want to start a new simulation run in the same folder structure and want to overwrite existing results, as failing to do so may lead to conflicts deleting old logs.

## Trying out ASSUME and the provided Examples

To ease your way into ASSUME we provided some examples and tutorials.
The former are helpful if you would like to get an impression of how ASSUME works and the latter introduce you into the development of ASSUME.

### The Tutorials

The tutorials work completely detached from your own machine on google colab.
They provide code snippets and task that show you, how you can work with the software package one your own.
We have multiple tutorials prepared, e.g. one for introducing a new unit and three for getting reinforcement learning ready on ASSUME.

How to configure a new unit in ASSUME?

[![Open Learning Tutorial in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/assume-framework/assume/blob/main/examples/notebooks/03_custom_unit_example.ipynb)

How to change and adapt reinforcement learning algorithms in ASSUME?

[![Open Learning Tutorial in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/assume-framework/assume/blob/main/examples/notebooks/04a_reinforcement_learning_algorithm_example.ipynb)

How to use reinforcement learning for new market participants in ASSUME?

- Power plant unit: [![Open Learning Tutorial in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/assume-framework/assume/blob/main/examples/notebooks/04b_reinforcement_learning_example.ipynb)

- Storage unit: [![Open Learning Tutorial in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/assume-framework/assume/blob/main/examples/notebooks/04c_reinforcement_learning_storage_example.ipynb)


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

If you have installed Docker and set up the Docker Compose file previously, you can select 'timescale' in `examples.py` before running the simulation.
This will save the simulation results in a Timescale database, and you can access the Dashboard at `http://localhost:3000`.

- Using the CLI to run simulations:

```bash
assume -s example_01b -db "postgresql://assume:assume@localhost:5432/assume"
```

For additional CLI options, run `assume -h`.

## Development

[The Contribution Guidelines explain how to setup your development environment and contribute to the project.](./CONTRIBUTING.md#development-setup)

## Creating Documentation

[See the Contribution Guidelines on how to build the docs for ASSUME.](./CONTRIBUTING.md#building-documentation)

## Contributors and Funding

The project is developed by a collaborative team of researchers from INATECH at the University of Freiburg, IISM at Karlsruhe Institute of Technology, Fraunhofer Institute for Systems and Innovation Research, Fraunhofer Institution for Energy Infrastructures and Geothermal Energy, and FH Aachen - University of Applied Sciences. Each contributor brings valuable expertise in electricity market modeling, deep reinforcement learning, demand side flexibility, and infrastructure modeling.

ASSUME is funded by the Federal Ministry for Economic Affairs and Climate Action (BMWK).
We are grateful for their support in making this project possible.

## Citing ASSUME

If you use **ASSUME** in your research, we would appreciate it if you cite the following paper:

* Nick Harder, Kim K. Miskiw, Manish Khanra, Florian Maurer, Parag Patil, Ramiz Qussous, Christof Weinhardt, Marian Klobasa, Mario Ragwitz, Anke Weidlich,
[ASSUME: An agent-based simulation framework for exploring electricity market dynamics with reinforcement learning](https://www.sciencedirect.com/science/article/pii/S2352711025001438),  published in [*SoftwareX*](https://www.sciencedirect.com/journal/softwarex), Volume 30, 2025, Article 102176.

Please use the following BibTeX to cite our work:

```bibtex
@article{ASSUME,
  title = {{ASSUME: An agent-based simulation framework for exploring electricity market dynamics with reinforcement learning}},
  author = {Harder, Nick and Miskiw, Kim K and Khanra, Manish and Maurer, Florian and Patil, Parag and Qussous, Ramiz and Weinhardt, Christof and Klobasa, Marian and Ragwitz, Mario and Weidlich, Anke},
  journal = {SoftwareX},
  volume = {30},
  pages = {102176},
  year = {2025},
  issn = {2352-7110},
  doi = {10.1016/j.softx.2025.102176},
  url = {https://www.sciencedirect.com/science/article/pii/S2352711025001438},
  keywords = {Electricity markets, Python, Reinforcement learning, Agent-based modeling}
}
```

If you want to cite a specific version of ASSUME, all releases are archived on Zenodo with version-specific DOIs:

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15065164.svg)](https://doi.org/10.5281/zenodo.15065164)

## License

Copyright 2022-2025 [ASSUME developers](https://assume.readthedocs.io/en/latest/developers.html).

ASSUME is licensed under the [GNU Affero General Public License v3.0](./LICENSES/AGPL-3.0-or-later.txt). This license is a strong copyleft license that requires that any derivative work be licensed under the same terms as the original work. It is approved by the [Open Source Initiative](https://opensource.org/licenses/AGPL-3.0).
