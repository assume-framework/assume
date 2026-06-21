<!--
SPDX-FileCopyrightText: ASSUME Developers

SPDX-License-Identifier: AGPL-3.0-or-later
-->

# ASSUME-MPEC
# Bilevel Optimization for Electricity Market Dynamics

## Overview

This folder extends the files and codes used in the paper titled **"How Satisfactory Can Deep Reinforcement Learning Methods Simulate Electricity Market Dynamics? Benchmarking via Bi-level Optimization"** presented at the DACH+ Energy Informatics 2024 conference. [Link to the paper.](https://energy.acm.org/eir/how-satisfactory-can-deep-reinforcement-learning-methods-simulate-electricity-market-dynamics-bechmarking-via-bi-level-optimization/)


## Convergence Status

### Working: Simple scenarios (02b)

The MPEC formulation converges reliably for simpler test cases like **example_02b** (few generators, single demand bid, no ramping constraints). Even with high MIP gaps (up to 10,000%), the UC re-solve validation produces sensible market clearing prices. The bilevel structure works as intended: the leader finds profitable bid multipliers k, and the UC re-solve confirms the resulting prices are plausible.

### Problematic: Complex scenarios (02a — elastic demand + ramping + renewables)

For the full **example_02a** scenario (7 generators, 10 elastic demand bids, ramping constraints, wind with time-varying availability), the MPEC does **not converge reliably**:

- **Root cause**: The bilinear term `k × mc × g` makes the problem non-convex quadratic. Gurobi's LP relaxation is very weak for this structure — the BestBound remains orders of magnitude away from the Incumbent.
- **MIP gap**: Typically 200–500% even after 10+ minutes of solve time. The gap is driven by the `big_w × duality_gap` penalty term in the objective, which dominates the bound.
- **Results**: Sometimes the solver finds good incumbents early (within seconds), sometimes not. The UC re-solve MCP is plausible in some hours but nonsensical in others. Results are more random than reliable.

#### What was tried and didn't solve it

| Approach | Outcome |
|----------|---------|
| Per-constraint Big-M values | LP relaxation barely tightens; still ~470,000% gap |
| Tighter global Big-M (10,000 vs. 10e6) | Reduces gap from 835% to ~270%, but not enough |
| Reducing duality gap weight (`big_w=1` vs. 10,000) | Helps convergence but weakens dual enforcement |
| UC warmstart for MPEC | Provides feasible starting point, marginal speed improvement |
| Linearized formulation (`find_optimal_dispatch_linearized`) | Binary expansion of k; still ~835% gap, much slower |

#### Known bug (fixed)

`duality_gap_part_1_rule` was missing the `model.g[gen, t]` factor for the `opt_gen` branch. This made the effective `big_w` penalty ~750× weaker than intended. With the bug present, the solver converged easily (the penalty was too weak to matter). After fixing, convergence became much harder — confirming that the strong duality enforcement is what makes this problem hard.

## File Overview

### Core files

| File | Description |
|------|-------------|
| `bilevel_opt.py` | All MPEC formulations (see table below) |
| `uc_problem.py` | Unit-commitment market clearing (LP relaxation for dual prices) |
| `utils.py` | Data loading, profit calculation, and helper utilities |

### Solve scripts

| File | Scenario | Description |
|------|----------|-------------|
| `solve_all_learning_units.py` | future_markets / 02e | Batch-solve MPEC + UC re-solve (original paper scenario, with storage) |
| `solve_all_learning_units_02a.py` | 02a | Batch-solve for elastic demand + ramping + renewables (with UC warmstart and merit order plots) |

### Notebooks

| File | Description |
|------|-------------|
| `12_eval_futur_markets_data.ipynb` | Export simulation data to `mpec_input_data/` |
| `12_eval_02a_elasticDemand_EE_Ramp.ipynb` | Export + analysis for 02a scenario (interactive merit order plots) |
| `12_eval_02b_data.ipynb` | Export data for 02b scenario |
| `12_eval_02b_final.ipynb` | MPEC variant testing on 02b (quadratic vs. linearised, single vs. multi demand) |
| `12_eval_02e_data.ipynb` | Notebook for example_02e scenario |
| `12_eval_02e_data_simplified.ipynb` | Simplified test-case notebook |
| `12_eval_03a_data.ipynb` | Export data for 03a scenario |
| `12_eval_test_functions.ipynb` | Function-level testing notebook |
| `clean_slate_future_markets_paper.ipynb` | Full paper reproduction notebook |

### Data directories

| Directory | Scenario | Contents |
|-----------|----------|----------|
| `mpec_input_data/` | future_markets / 02e | gens_df, demand_df, k_values_df, availability_df, dispatch_df, storage |
| `mpec_input_data_02a/` | 02a | Same structure, no storage, 10 elastic demand bids |
| `results/` | future_markets | Result CSVs + plots |
| `results_02a/` | 02a | Result CSVs + merit order plots |

## MPEC Formulations

| Function | Leader Decision | Storage Handling | Notes |
|----------|-----------------|------------------|-------|
| `find_optimal_dispatch_linearized` | k ∈ {discrete steps} | Not supported | Binary expansion of k, slower |
| `find_optimal_dispatch_quadratic` | k ∈ [1, k_max] | In lower level | Original quadratic formulation |
| `find_optimal_dispatch_quadratic_with_storage` | k ∈ [1, k_max] | Storage as decision variable | Does not converge in reasonable time |
| `find_optimal_dispatch_storage_leader_quadratic` | k ∈ [1, k_max] | Storage as leader variable | Produces nonsensical results — bidding strategy violates assumptions from Li et al. |
| `find_optimal_dispatch_quadratic_fixed_storage` | k ∈ [1, k_max] | Fixed (folded into demand) | **Recommended** — exogenous storage |

All formulations are in `bilevel_opt.py`.

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
12_eval_*_data.ipynb               ← exports ALL data (no sampling)
        │
        ▼
  mpec_input_data[_02a]/           ← gens_df, demand_df, k_values_df,
        │                            availability_df, dispatch_df
        ▼
solve_all_learning_units[_02a].py  ← optional date-range filter (SOLVE_START/END)
        │                            MPEC → UC re-solve for each learning unit
        ▼
    results[_02a]/                 ← CSVs + plots (MCP timeseries + merit order)
```

## Key Concepts

- **k-multiplier**: The leader's decision variable. Effective bid price = mc × k, where k ∈ [1, k_max].
- **Hat variables (λ̂)**: Relaxed KKT dual variables. The duality gap penalty `big_w × (primal − dual)` drives λ̂ toward the true dual λ.
- **Big-M complementarity**: KKT complementary slackness conditions are linearised using big-M constraints.
- **UC re-solve**: After MPEC finds optimal k-values, a unit-commitment problem is solved with fixed binaries to obtain accurate market clearing prices.
- **Fixed storage dispatch**: Storage dispatch from the original simulation is subtracted from demand, making storage exogenous to the MPEC.
- **UC warmstart** (02a only): A baseline UC solve (k=1) provides a feasible starting point for the MPEC solver's primal variables (g, u, c_up, c_down, λ).

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

## Acknowledgments

This work was conducted in the context of the project **"ASSUME: Agent-Based Electricity Markets Simulation Toolbox,"** funded by the German Federal Ministry for Economic Affairs and Energy under grant number BMWK 03EI1052A.
