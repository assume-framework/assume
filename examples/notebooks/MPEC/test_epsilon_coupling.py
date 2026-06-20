"""
Test: Tight variable bounds + big_M=10000 + epsilon coupling |lambda_hat - lambda_| <= eps.
Hypothesis: tight bounds enable fast LP relaxation, big_M=10000 allows good complementarity,
epsilon coupling prevents mcp_hat=3000 artefact.
"""
import time, pandas as pd, numpy as np
import pyomo.environ as pyo
from pyomo.opt import SolverFactory

data_dir = 'mpec_input_data_02a'
gens_df = pd.read_csv(f'{data_dir}/gens_df.csv', index_col=0)
demand_df = pd.read_csv(f'{data_dir}/demand_df.csv', index_col=0, parse_dates=True)
k_values_df = pd.read_csv(f'{data_dir}/k_values_df.csv', index_col=0, parse_dates=True).fillna(1.0)
availability_df = pd.read_csv(f'{data_dir}/availability_df.csv', index_col=0, parse_dates=True)

mask = (k_values_df.index >= '2019-03-16') & (k_values_df.index < '2019-03-17')
k_values_df = k_values_df[mask]; demand_df = demand_df[mask]; availability_df = availability_df[mask]
demand_df = demand_df.reset_index(drop=True)
k_values_df = k_values_df.reset_index(drop=True).drop(columns=['date'], errors='ignore')
availability_df = availability_df.reset_index(drop=True)
vol_cols = [c for c in demand_df.columns if c.startswith('volume_')]
demand_df[vol_cols] = demand_df[vol_cols].abs().clip(lower=0.01)
demand_df = demand_df.drop(columns=['date'], errors='ignore')
n_demand_bids = len(vol_cols)
gens_df = gens_df.set_index('unit') if 'unit' in gens_df.columns else gens_df
gen_names = list(gens_df.index)
k_values_df = k_values_df[[c for c in k_values_df.columns if c in gen_names]]
availability_df = availability_df[[c for c in availability_df.columns if c in gen_names]]
mc_df = pd.DataFrame({g: gens_df.at[g, 'mc'] for g in gen_names}, index=demand_df.index)


def solve_eps_coupling(opt_gen, k_max, time_limit=600, big_w=10000, big_M=10000,
                       dual_bound=3500, epsilon=50):
    """MPEC with tight variable bounds, big_M for complementarity, and epsilon coupling."""
    model = pyo.ConcreteModel()
    model.time = pyo.Set(initialize=demand_df.index)
    model.gens = pyo.Set(initialize=gens_df.index)
    model.demand_bids = pyo.Set(initialize=np.arange(1, n_demand_bids + 1))

    # PRIMAL with tight bounds
    model.g = pyo.Var(model.gens, model.time, within=pyo.NonNegativeReals,
                      bounds=lambda m, i, t: (0, gens_df.at[i, 'g_max']))
    model.d = pyo.Var(model.time, model.demand_bids, within=pyo.NonNegativeReals,
                      bounds=lambda m, t, n: (0, demand_df.at[t, f'volume_{n}']))
    model.c_up = pyo.Var(model.gens, model.time, within=pyo.NonNegativeReals)
    model.c_down = pyo.Var(model.gens, model.time, within=pyo.NonNegativeReals)
    model.k = pyo.Var(model.time, bounds=(1, k_max), within=pyo.NonNegativeReals)
    model.u = pyo.Var(model.gens, model.time, within=pyo.Binary)

    # TWO price variables but epsilon-coupled
    model.lambda_ = pyo.Var(model.time, within=pyo.Reals, bounds=(-500, 3100))
    model.lambda_hat = pyo.Var(model.time, within=pyo.Reals, bounds=(-500, 3100))

    # Non-hat duals with tight bounds
    model.mu_max = pyo.Var(model.gens, model.time, bounds=(0, dual_bound))
    model.mu_min = pyo.Var(model.gens, model.time, bounds=(0, dual_bound))
    model.nu_max = pyo.Var(model.time, model.demand_bids, bounds=(0, dual_bound))
    model.pi_u = pyo.Var(model.gens, model.time, bounds=(0, dual_bound))
    model.pi_d = pyo.Var(model.gens, model.time, bounds=(0, dual_bound))
    model.sigma_u = pyo.Var(model.gens, model.time, bounds=(0, 1))
    model.sigma_d = pyo.Var(model.gens, model.time, bounds=(0, 1))
    model.psi_max = pyo.Var(model.gens, model.time, bounds=(0, 5000))

    # Hat duals with tight bounds
    model.mu_max_hat = pyo.Var(model.gens, model.time, bounds=(0, dual_bound))
    model.mu_min_hat = pyo.Var(model.gens, model.time, bounds=(0, dual_bound))
    model.nu_max_hat = pyo.Var(model.time, model.demand_bids, bounds=(0, dual_bound))
    model.nu_min_hat = pyo.Var(model.time, model.demand_bids, bounds=(0, dual_bound))
    model.pi_u_hat = pyo.Var(model.gens, model.time, bounds=(0, dual_bound))
    model.pi_d_hat = pyo.Var(model.gens, model.time, bounds=(0, dual_bound))

    # Complementarity binaries
    model.mu_max_hat_binary = pyo.Var(model.gens, model.time, within=pyo.Binary)
    model.nu_max_hat_b = pyo.Var(model.time, model.demand_bids, within=pyo.Binary)
    model.nu_min_hat_b = pyo.Var(model.time, model.demand_bids, within=pyo.Binary)
    model.pi_u_hat_b = pyo.Var(model.gens, model.time, within=pyo.Binary)
    model.pi_d_hat_b = pyo.Var(model.gens, model.time, within=pyo.Binary)

    # === EPSILON COUPLING ===
    model.eps_ub = pyo.Constraint(model.time,
        rule=lambda m, t: m.lambda_hat[t] <= m.lambda_[t] + epsilon)
    model.eps_lb = pyo.Constraint(model.time,
        rule=lambda m, t: m.lambda_hat[t] >= m.lambda_[t] - epsilon)

    # === OBJECTIVE ===
    def primary_obj(model):
        return sum(
            model.lambda_hat[t] * model.g[opt_gen, t]
            - mc_df.at[t, opt_gen] * model.g[opt_gen, t]
            - model.c_up[opt_gen, t] - model.c_down[opt_gen, t]
            for t in model.time)

    def duality_gap_primal(model):
        expr = sum(
            (mc_df.at[t, gen] * model.k[t] * model.g[gen, t] + model.c_up[gen, t] + model.c_down[gen, t])
            if gen == opt_gen else
            (k_values_df.at[t, gen] * mc_df.at[t, gen] * model.g[gen, t] + model.c_up[gen, t] + model.c_down[gen, t])
            for gen in model.gens for t in model.time)
        expr -= sum(demand_df.at[t, f'price_{n}'] * model.d[t, n]
                    for t in model.time for n in model.demand_bids)
        return expr

    def duality_gap_dual(model):
        expr = -sum(model.nu_max[t, n] * demand_df.at[t, f'volume_{n}']
                    for t in model.time for n in model.demand_bids)
        expr -= sum(model.pi_u[i, t] * gens_df.at[i, 'r_up'] for i in model.gens for t in model.time)
        expr -= sum(model.pi_d[i, t] * gens_df.at[i, 'r_down'] for i in model.gens for t in model.time)
        expr -= sum(model.pi_u[i, 0] * gens_df.at[i, 'g_0'] for i in model.gens)
        expr += sum(model.pi_d[i, 0] * gens_df.at[i, 'g_0'] for i in model.gens)
        expr -= sum(model.sigma_u[i, 0] * gens_df.at[i, 'k_up'] * gens_df.at[i, 'u_0'] for i in model.gens)
        expr += sum(model.sigma_d[i, 0] * gens_df.at[i, 'k_down'] * gens_df.at[i, 'u_0'] for i in model.gens)
        expr -= sum(model.psi_max[i, t] for i in model.gens for t in model.time)
        return expr

    model.objective = pyo.Objective(
        expr=lambda m: primary_obj(m) - big_w * (duality_gap_primal(m) - duality_gap_dual(m)),
        sense=pyo.maximize)

    # === PRIMAL CONSTRAINTS ===
    model.balance = pyo.Constraint(model.time,
        rule=lambda m, t: sum(m.d[t, n] for n in m.demand_bids) == sum(m.g[i, t] for i in m.gens))
    model.g_max_c = pyo.Constraint(model.gens, model.time,
        rule=lambda m, i, t: m.g[i, t] <= gens_df.at[i, 'g_max'] * availability_df.at[t, i] * m.u[i, t])
    model.d_max_c = pyo.Constraint(model.time, model.demand_bids,
        rule=lambda m, t, n: m.d[t, n] <= demand_df.at[t, f'volume_{n}'])

    def ru_rule(m, i, t):
        prev = gens_df.at[i, 'g_0'] if t == 0 else m.g[i, t - 1]
        return m.g[i, t] - prev <= gens_df.at[i, 'r_up']
    model.ru_max = pyo.Constraint(model.gens, model.time, rule=ru_rule)
    def rd_rule(m, i, t):
        prev = gens_df.at[i, 'g_0'] if t == 0 else m.g[i, t - 1]
        return prev - m.g[i, t] <= gens_df.at[i, 'r_down']
    model.rd_max = pyo.Constraint(model.gens, model.time, rule=rd_rule)
    def cup_rule(m, i, t):
        prev_u = gens_df.at[i, 'u_0'] if t == 0 else m.u[i, t - 1]
        return m.c_up[i, t] >= (m.u[i, t] - prev_u) * gens_df.at[i, 'k_up']
    model.cup_c = pyo.Constraint(model.gens, model.time, rule=cup_rule)
    def cdn_rule(m, i, t):
        prev_u = gens_df.at[i, 'u_0'] if t == 0 else m.u[i, t - 1]
        return m.c_down[i, t] >= (prev_u - m.u[i, t]) * gens_df.at[i, 'k_down']
    model.cdn_c = pyo.Constraint(model.gens, model.time, rule=cdn_rule)

    # === DUAL FEASIBILITY (non-hat) ===
    def gen_dual(m, i, t):
        k_term = m.k[t] if i == opt_gen else k_values_df.at[t, i]
        pi_u_next = 0 if t == m.time.at(-1) else m.pi_u[i, t + 1]
        pi_d_next = 0 if t == m.time.at(-1) else m.pi_d[i, t + 1]
        return (k_term * mc_df.at[t, i] - m.lambda_[t] + m.mu_max[i, t] - m.mu_min[i, t]
                + m.pi_u[i, t] - pi_u_next - m.pi_d[i, t] + pi_d_next == 0)
    model.gen_dual = pyo.Constraint(model.gens, model.time, rule=gen_dual)

    def status_dual(m, i, t):
        if t != m.time.at(-1):
            return (-m.mu_max[i, t] * gens_df.at[i, 'g_max'] * availability_df.at[t, i]
                    + (m.sigma_u[i, t] - m.sigma_u[i, t + 1]) * gens_df.at[i, 'k_up']
                    - (m.sigma_d[i, t] - m.sigma_d[i, t + 1]) * gens_df.at[i, 'k_down']
                    + m.psi_max[i, t] >= 0)
        return (-m.mu_max[i, t] * gens_df.at[i, 'g_max'] * availability_df.at[t, i]
                + m.sigma_u[i, t] * gens_df.at[i, 'k_up']
                - m.sigma_d[i, t] * gens_df.at[i, 'k_down']
                + m.psi_max[i, t] >= 0)
    model.status_dual_c = pyo.Constraint(model.gens, model.time, rule=status_dual)

    model.demand_dual = pyo.Constraint(model.time, model.demand_bids,
        rule=lambda m, t, n: -demand_df.at[t, f'price_{n}'] + m.lambda_[t] + m.nu_max[t, n] >= 0)

    # === HAT-KKT STATIONARITY ===
    def kkt_gen(m, i, t):
        k_term = m.k[t] if i == opt_gen else k_values_df.at[t, i]
        puh_next = 0 if t == m.time.at(-1) else m.pi_u_hat[i, t + 1]
        pdh_next = 0 if t == m.time.at(-1) else m.pi_d_hat[i, t + 1]
        return (k_term * mc_df.at[t, i] - m.lambda_hat[t] + m.mu_max_hat[i, t] - m.mu_min_hat[i, t]
                + m.pi_u_hat[i, t] - puh_next - m.pi_d_hat[i, t] + pdh_next == 0)
    model.kkt_gen = pyo.Constraint(model.gens, model.time, rule=kkt_gen)
    model.kkt_demand = pyo.Constraint(model.time, model.demand_bids,
        rule=lambda m, t, n: -demand_df.at[t, f'price_{n}'] + m.lambda_hat[t] + m.nu_max_hat[t, n] - m.nu_min_hat[t, n] == 0)

    # === COMPLEMENTARITY ===
    model.mu_hat_b1 = pyo.Constraint(model.gens, model.time,
        rule=lambda m, i, t: m.g[i, t] - gens_df.at[i, 'g_max'] * availability_df.at[t, i] * m.u[i, t]
                              <= gens_df.at[i, 'g_max'] * (1 - m.mu_max_hat_binary[i, t]))
    model.mu_hat_b2 = pyo.Constraint(model.gens, model.time,
        rule=lambda m, i, t: m.g[i, t] - gens_df.at[i, 'g_max'] * availability_df.at[t, i] * m.u[i, t]
                              >= -gens_df.at[i, 'g_max'] * (1 - m.mu_max_hat_binary[i, t]))
    model.mu_hat_b3 = pyo.Constraint(model.gens, model.time,
        rule=lambda m, i, t: m.mu_max_hat[i, t] <= big_M * m.mu_max_hat_binary[i, t])

    model.nu_hat_b1 = pyo.Constraint(model.time, model.demand_bids,
        rule=lambda m, t, n: m.d[t, n] - demand_df.at[t, f'volume_{n}']
                              <= demand_df.at[t, f'volume_{n}'] * (1 - m.nu_max_hat_b[t, n]))
    model.nu_hat_b2 = pyo.Constraint(model.time, model.demand_bids,
        rule=lambda m, t, n: m.d[t, n] - demand_df.at[t, f'volume_{n}']
                              >= -demand_df.at[t, f'volume_{n}'] * (1 - m.nu_max_hat_b[t, n]))
    model.nu_hat_b3 = pyo.Constraint(model.time, model.demand_bids,
        rule=lambda m, t, n: m.nu_max_hat[t, n] <= big_M * m.nu_max_hat_b[t, n])

    model.numinh_b1 = pyo.Constraint(model.time, model.demand_bids,
        rule=lambda m, t, n: m.d[t, n] <= demand_df.at[t, f'volume_{n}'] * (1 - m.nu_min_hat_b[t, n]))
    model.numinh_b3 = pyo.Constraint(model.time, model.demand_bids,
        rule=lambda m, t, n: m.nu_min_hat[t, n] <= big_M * m.nu_min_hat_b[t, n])

    def piu1(m, i, t):
        rb = gens_df.at[i, 'g_max']
        prev = gens_df.at[i, 'g_0'] if t == 0 else m.g[i, t - 1]
        return m.g[i, t] - prev - gens_df.at[i, 'r_up'] <= rb * (1 - m.pi_u_hat_b[i, t])
    def piu2(m, i, t):
        rb = gens_df.at[i, 'g_max']
        prev = gens_df.at[i, 'g_0'] if t == 0 else m.g[i, t - 1]
        return m.g[i, t] - prev - gens_df.at[i, 'r_up'] >= -rb * (1 - m.pi_u_hat_b[i, t])
    model.piuh1 = pyo.Constraint(model.gens, model.time, rule=piu1)
    model.piuh2 = pyo.Constraint(model.gens, model.time, rule=piu2)
    model.piuh3 = pyo.Constraint(model.gens, model.time,
        rule=lambda m, i, t: m.pi_u_hat[i, t] <= big_M * m.pi_u_hat_b[i, t])

    def pid1(m, i, t):
        rb = gens_df.at[i, 'g_max']
        prev = gens_df.at[i, 'g_0'] if t == 0 else m.g[i, t - 1]
        return prev - m.g[i, t] - gens_df.at[i, 'r_down'] <= rb * (1 - m.pi_d_hat_b[i, t])
    def pid2(m, i, t):
        rb = gens_df.at[i, 'g_max']
        prev = gens_df.at[i, 'g_0'] if t == 0 else m.g[i, t - 1]
        return prev - m.g[i, t] - gens_df.at[i, 'r_down'] >= -rb * (1 - m.pi_d_hat_b[i, t])
    model.pidh1 = pyo.Constraint(model.gens, model.time, rule=pid1)
    model.pidh2 = pyo.Constraint(model.gens, model.time, rule=pid2)
    model.pidh3 = pyo.Constraint(model.gens, model.time,
        rule=lambda m, i, t: m.pi_d_hat[i, t] <= big_M * m.pi_d_hat_b[i, t])

    # === SOLVE ===
    instance = model.create_instance()
    solver = SolverFactory('gurobi')
    solver.options['NonConvex'] = 2
    results = solver.solve(instance, options={
        'LogToConsole': 1, 'TimeLimit': time_limit, 'MIPGap': 0.05,
        'MIPFocus': 1, 'Presolve': 2, 'DualReductions': 0,
    }, tee=True)

    tc = results.solver.termination_condition
    print(f'\nStatus: {tc}')

    if tc in (pyo.TerminationCondition.infeasible, pyo.TerminationCondition.infeasibleOrUnbounded):
        return None

    ti = demand_df.index
    mcp = [float(instance.lambda_[t].value) for t in ti]
    mcp_hat = [float(instance.lambda_hat[t].value) for t in ti]
    k_vals = [float(instance.k[t].value) for t in ti]
    gen_vals = [float(instance.g[opt_gen, t].value) for t in ti]
    demand_vals = [sum(float(instance.d[t, n].value) for n in instance.demand_bids) for t in ti]

    return {
        'mcp': mcp, 'mcp_hat': mcp_hat, 'k': k_vals,
        f'gen_{opt_gen}': gen_vals, 'demand': demand_vals
    }


if __name__ == '__main__':
    configs = [
        # (opt_gen, k_max, big_M, dual_bound, epsilon, description)
        ('pp_3', 99, 10000, 3500, 50, 'tight bounds + big_M=10k + eps=50'),
        ('pp_3', 99, 10000, 3500, 10, 'tight bounds + big_M=10k + eps=10'),
        ('pp_3', 99, 3500, 3500, 50, 'tight bounds + big_M=3.5k + eps=50'),
    ]

    for opt_gen, k_max, big_M, dual_bound, eps, desc in configs:
        print(f'\n{"=" * 70}')
        print(f'{desc}: {opt_gen}, mc={gens_df.at[opt_gen, "mc"]:.2f}')
        print(f'{"=" * 70}')

        t0 = time.time()
        res = solve_eps_coupling(opt_gen, k_max, time_limit=300,
                                  big_M=big_M, dual_bound=dual_bound, epsilon=eps)
        elapsed = time.time() - t0

        if res:
            mcp = np.array(res['mcp'])
            mcp_hat = np.array(res['mcp_hat'])
            k = np.array(res['k'])
            g = np.array(res[f'gen_{opt_gen}'])
            d = np.array(res['demand'])
            print(f'\nTime: {elapsed:.1f}s')
            print(f'MCP (lambda_):   {mcp.min():.2f} to {mcp.max():.2f}, mean={mcp.mean():.2f}')
            print(f'MCP_hat:         {mcp_hat.min():.2f} to {mcp_hat.max():.2f}, mean={mcp_hat.mean():.2f}')
            print(f'|hat - lambda|:  max={np.abs(mcp_hat - mcp).max():.2f}')
            print(f'K: {k.min():.4f} to {k.max():.4f}, mean={k.mean():.4f}')
            print(f'Gen: {g.min():.0f} to {g.max():.0f}')
            print(f'Demand: {d.min():.0f} to {d.max():.0f}')
            profit = sum((mcp_hat - gens_df.at[opt_gen, 'mc']) * g)
            print(f'Profit (hat): {profit:.0f} EUR/day')
            profit_real = sum((mcp - gens_df.at[opt_gen, 'mc']) * g)
            print(f'Profit (real): {profit_real:.0f} EUR/day')
        else:
            print(f'INFEASIBLE ({elapsed:.1f}s)')
