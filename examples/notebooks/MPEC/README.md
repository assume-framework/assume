# ASSUME-MPEC
# Bilevel Optimization for Electricity Market Dynamics

## Overview

This repository contains the files and codes used in the paper titled **"How Satisfactory Can Deep Reinforcement Learning Methods Simulate Electricity Market Dynamics? Benchmarking via Bi-level Optimization"** presented at the DACH+ Energy Informatics 2024 conference. [Link to the paper.](https://energy.acm.org/eir/how-satisfactory-can-deep-reinforcement-learning-methods-simulate-electricity-market-dynamics-bechmarking-via-bi-level-optimization/)

## Abstract

Various factors make electricity markets increasingly complex, making their analysis challenging. This complexity demands advanced analytical tools to manage and understand market dynamics. This paper explores the application of deep reinforcement learning (DRL) and bi-level optimization models to analyze and simulate electricity markets. We introduce a bi-level optimization framework incorporating realistic market constraints, such as non-convex operational characteristics and binary decision variables, to establish an upper-bound benchmark for evaluating the performance of DRL algorithms. 

The results confirm that DRL methods do not reach the theoretical upper bounds set by the bi-level models, thereby confirming the effectiveness of the proposed model in providing a clear performance target for DRL. This benchmarking approach demonstrates DRL's current capabilities and limitations in complex market environments but also aids in developing more effective DRL strategies by providing clear, quantifiable targets for improvement. 

The proposed method can also identify the information gap cost since DRL methods operate under more realistic conditions than optimization techniques, given that they don't need to assume complete knowledge about the system. This study thus provides a foundation for future research to enhance market understanding and possibly its efficiency in the face of increasing complexity in the electricity market. Our methodology's effectiveness is further validated through a large-scale case study involving 150 power plants, demonstrating its scalability and applicability to real-world scenarios.

## Repository Contents

- **bilevel_opt.py**: Implements the proposed method for benchmarking.
- **utils.py**: Provides utility functions to support the main models.
- **uc_problem.py**: Contains the market clearing problem formulation.

- **Environment Configuration**: An `environment.yaml` file for setting up the Conda environment.

## Getting Started

To get started with the code in this repository, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/nick-harder/ASSUME-MPEC.git
   cd ASSUME-MPEC
   ```

2. Create the Conda environment:
   ```bash
   conda env create -f environment.yaml
   conda activate your_environment_name
   ```

3. Run the main script:
   ```bash
   python bilevel_opt.py
   ```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

This work was conducted in the context of the project **"ASSUME: Agent-Based Electricity Markets Simulation Toolbox,"** funded by the German Federal Ministry for Economic Affairs and Energy under grant number BMWK 03EI1052A. It was also supported by the Scientific Society in Freiburg im Breisgau and the German Academic Exchange Service, which partially funded the research stay at the Technical University of Denmark.