#!/usr/bin/env python3
"""Check if data exists for the current simulation"""

import pandas as pd
from sqlalchemy import create_engine

# Database connection
db_uri = "postgresql://assume:assume@localhost:5432/assume"
engine = create_engine(db_uri)

simulation_id = "2_nodes_learning_single_diesel"

# Check if any data exists for this simulation
query = f"""
SELECT 
    COUNT(*) as total_rows,
    COUNT(DISTINCT episode) as episodes,
    MIN(datetime) as min_date,
    MAX(datetime) as max_date
FROM rl_params 
WHERE simulation = '{simulation_id}'
"""

print(f"Checking for simulation: {simulation_id}")
df = pd.read_sql(query, engine)
print(df)

if df['total_rows'][0] == 0:
    print(f"\n❌ No data found for '{simulation_id}'")
    print("\nThis means data is NOT being written to the database during your simulation.")
    print("\nPossible causes:")
    print("1. The database connection is not being used (check if db_uri is passed to World)")
    print("2. The output role is not set up")
    print("3. There's an error during data writing that's being silently caught")
    print("4. Data is being written to a different database")
    
    # Check what simulations exist
    sims = pd.read_sql("SELECT DISTINCT simulation FROM rl_params LIMIT 10", engine)
    print(f"\nSimulations in database: {sims['simulation'].tolist()}")
else:
    print(f"\n✅ Data exists for '{simulation_id}'!")
    print(f"   Episodes: {df['episodes'][0]}")
    print(f"   Date range: {df['min_date'][0]} to {df['max_date'][0]}")
