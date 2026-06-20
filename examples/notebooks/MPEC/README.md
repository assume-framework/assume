<!--
SPDX-FileCopyrightText: ASSUME Developers

SPDX-License-Identifier: AGPL-3.0-or-later
-->

# ASSUME-MPEC
# Bilevel Optimization for Electricity Market Dynamics

## Overview

This folder entends the files and codes used in the paper titled **"How Satisfactory Can Deep Reinforcement Learning Methods Simulate Electricity Market Dynamics? Benchmarking via Bi-level Optimization"** presented at the DACH+ Energy Informatics 2024 conference. [Link to the paper.](https://energy.acm.org/eir/how-satisfactory-can-deep-reinforcement-learning-methods-simulate-electricity-market-dynamics-bechmarking-via-bi-level-optimization/)

## Abstract

Various factors make electricity markets increasingly complex, making their analysis challenging. This complexity demands advanced analytical tools to manage and understand market dynamics. This paper explores the application of deep reinforcement learning (DRL) and bi-level optimization models to analyze and simulate electricity markets. We introduce a bi-level optimization framework incorporating realistic market constraints, such as non-convex operational characteristics and binary decision variables, to establish an upper-bound benchmark for evaluating the performance of DRL algorithms.

The results confirm that DRL methods do not reach the theoretical upper bounds set by the bi-level models, thereby confirming the effectiveness of the proposed model in providing a clear performance target for DRL. This benchmarking approach demonstrates DRL's current capabilities and limitations in complex market environments but also aids in developing more effective DRL strategies by providing clear, quantifiable targets for improvement.

The proposed method can also identify the information gap cost since DRL methods operate under more realistic conditions than optimization techniques, given that they don't need to assume complete knowledge about the system. This study thus provides a foundation for future research to enhance market understanding and possibly its efficiency in the face of increasing complexity in the electricity market. Our methodology's effectiveness is further validated through a large-scale case study involving 150 power plants, demonstrating its scalability and applicability to real-world scenarios.

## File Overview

| File | Description |
|------|-------------|
| `bilevel_opt.py` | All MPEC formulations (see table below) |
| `uc_problem.py` | Unit-commitment market clearing (LP relaxation for dual prices) |
| `utils.py` | Data loading, profit calculation, and helper utilities |
| `solve_all_learning_units.py` | Batch-solve MPEC + UC re-solve for all learning units |
| `12_eval_futur_markets_data.ipynb` | Export simulation data to `mpec_input_data/` |
| `12_eval_02e_data.ipynb` | Notebook for example_02e scenario |
| `12_eval_02e_data_simplified.ipynb` | Simplified test-case notebook |
| `clean_slate_future_markets_paper.ipynb` | Full paper reproduction notebook |

## MPEC Formulations

| Function | File | Leader Decision | Storage Handling | Notes |
|----------|------|-----------------|------------------|-------|
| `find_optimal_dispatch_linearized` | `bilevel_opt.py` | k ∈ {discrete steps} | Not supported | Binary expansion of k, slower |
| `find_optimal_dispatch_quadratic` | `bilevel_opt.py` | k ∈ [1, k_max] | In lower level | Original quadratic formulation |
| `find_optimal_dispatch_quadratic_with_storage` | `bilevel_opt.py` | k ∈ [1, k_max] | Storage as decision variable | Full storage co-optimisation, does not converge in reasonable time |
| `find_optimal_dispatch_storage_leader_quadratic` | `bilevel_opt.py` | k ∈ [1, k_max] | Storage as leader variable | Storage in upper level, produces super weird results because bidding startegy does not provide necessary sensible bids to align with assumption in Li et al.  |
| `find_optimal_dispatch_quadratic_fixed_storage` | `bilevel_opt.py` | k ∈ [1, k_max] | Fixed (folded into demand) | **Recommended** — exogenous storage |

### UC Re-Solve Functions

| Function | File | Description |
|----------|------|-------------|
| `solve_uc_problem` | `uc_problem.py` | LP-relaxed UC with fixed k-values → dual prices as MCPs |
| `solve_uc_problem_with_storage` | `uc_problem.py` | UC with storage as decision variable |

## Data Flow

```
ASSUME simulation DB
        │
        ▼
12_eval_futur_markets_data.ipynb   ← exports ALL data (no sampling)
        │
        ▼
  mpec_input_data/                 ← gens_df, demand_df, k_values_df,
        │                            availability_df, dispatch_df
        ▼
solve_all_learning_units.py        ← optional date-range filter (SOLVE_START/END)
        │                            MPEC → UC re-solve for each learning unit
        ▼
    results/                       ← CSVs + plots
```

## Getting Started

1. Set up the Conda environment:
   ```bash
   conda activate assume-framework
   ```

2. Export data from a simulation run (requires access to the simulation DB):
   ```bash
   jupyter notebook 12_eval_futur_markets_data.ipynb
   ```

3. Solve MPEC for all learning units:
   ```bash
   python solve_all_learning_units.py
   ```
   Edit `SOLVE_START` / `SOLVE_END` at the top of the script to select which dates to solve (set to `None` to solve all exported dates).

## Key Concepts

- **k-multiplier**: The leader's decision variable. Effective bid price = mc × k, where k ∈ [1, k_max].
- **Hat variables (λ̂)**: Relaxed KKT dual variables. The duality gap penalty `big_w × (primal − dual)` drives λ̂ toward the true dual λ.
- **Big-M complementarity**: KKT complementary slackness conditions are linearised using big-M constraints.
- **UC re-solve**: After MPEC finds optimal k-values, a unit-commitment problem is solved with fixed binaries to obtain accurate market clearing prices.
- **Fixed storage dispatch**: Storage dispatch from the original simulation is subtracted from demand, making storage exogenous to the MPEC.


## Acknowledgments

This work was conducted in the context of the project **"ASSUME: Agent-Based Electricity Markets Simulation Toolbox,"** funded by the German Federal Ministry for Economic Affairs and Energy under grant number BMWK 03EI1052A. 