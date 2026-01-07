#!/usr/bin/env python3
"""Test the actual query from tensorboard_logger"""

import pandas as pd
from sqlalchemy import create_engine

# Database connection
db_uri = "postgresql://assume:assume@localhost:5432/assume"
engine = create_engine(db_uri)

# Test parameters - adjust these to match your simulation
simulation_id = "2_nodes_learning_single_diesel"  # Update this!
episode = 1
evaluation_mode = False

print(f"Testing query for:")
print(f"  simulation_id: {simulation_id}")
print(f"  episode: {episode}")
print(f"  evaluation_mode: {evaluation_mode}")

# Get column names (PostgreSQL version)
query_columns = """
SELECT column_name FROM information_schema.columns
WHERE table_name = 'rl_params'
"""
columns_df = pd.read_sql(query_columns, engine)
column_names = columns_df["column_name"].tolist()
print(f"\nColumns in rl_params: {column_names}")

# Check for noise columns
noise_columns = [col for col in column_names if col.startswith("exploration_noise_")]
print(f"Noise columns found: {noise_columns}")

# Build the date function (PostgreSQL)
date_func = "TO_CHAR(datetime, 'YYYY-MM-DD')"

# Build query parts
query_parts = [
    f"{date_func} AS dt",
    "unit",
    "SUM(profit) AS profit",
    "SUM(reward) AS reward",
]

if "regret" in column_names:
    query_parts.append("SUM(regret) AS regret")

if noise_columns:
    noise_sql = ", ".join([f"AVG({col}) AS {col}" for col in noise_columns])
    query_parts.append(noise_sql)

# Build the query
query_sim = f"""
SELECT
    {", ".join(query_parts)}
FROM rl_params
WHERE episode = '{episode}'
AND simulation = '{simulation_id}'
AND evaluation_mode = {evaluation_mode}
GROUP BY {date_func}, unit
ORDER BY {date_func}
"""

print(f"\nQuery to execute:")
print(query_sim)

# Execute the query
print("\nExecuting query...")
try:
    df_sim = pd.read_sql(query_sim, engine)
    print(f"\nQuery result: {len(df_sim)} rows")
    print(df_sim.head(10))
    
    if len(df_sim) == 0:
        print("\n⚠ DataFrame is empty! Let's check what data exists...")
        
        # Check if data exists for this simulation
        check_query = f"""
        SELECT COUNT(*) as count 
        FROM rl_params 
        WHERE simulation = '{simulation_id}'
        """
        check_df = pd.read_sql(check_query, engine)
        print(f"\nTotal rows for simulation '{simulation_id}': {check_df['count'][0]}")
        
        # Check what simulations exist
        sims_query = "SELECT DISTINCT simulation FROM rl_params"
        sims_df = pd.read_sql(sims_query, engine)
        print(f"\nAvailable simulations in database:")
        for sim in sims_df['simulation']:
            print(f"  - {sim}")
        
        # Check episodes for this simulation
        ep_query = f"""
        SELECT DISTINCT episode 
        FROM rl_params 
        WHERE simulation = '{simulation_id}'
        ORDER BY episode
        """
        ep_df = pd.read_sql(ep_query, engine)
        if len(ep_df) > 0:
            print(f"\nAvailable episodes for '{simulation_id}': {ep_df['episode'].tolist()}")
        else:
            print(f"\n⚠ No data found for simulation '{simulation_id}'")
            
except Exception as e:
    print(f"\n✗ Error executing query: {e}")
    import traceback
    traceback.print_exc()
