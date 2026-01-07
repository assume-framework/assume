#!/usr/bin/env python3
"""Script to check if data exists in the database"""

import pandas as pd
from sqlalchemy import create_engine, inspect, text

# Update this with your actual database URI
# For TimescaleDB, it should be something like:
# db_uri = "postgresql://user:password@localhost:5432/assume"
# For SQLite, it would be:
# db_uri = "sqlite:///assume_db.db"

db_uri = "postgresql://assume:assume@localhost:5432/assume"  # Adjust as needed

print(f"Connecting to: {db_uri}")
engine = create_engine(db_uri)

# Check if rl_params table exists
inspector = inspect(engine)
tables = inspector.get_table_names()
print(f"\nAvailable tables: {tables}")

if "rl_params" in tables:
    print("\n✓ rl_params table exists")
    
    # Get column info
    columns = inspector.get_columns("rl_params")
    print(f"\nColumns in rl_params:")
    for col in columns:
        print(f"  - {col['name']} ({col['type']})")
    
    # Count total rows
    with engine.connect() as conn:
        result = conn.execute(text("SELECT COUNT(*) FROM rl_params"))
        count = result.scalar()
        print(f"\nTotal rows in rl_params: {count}")
        
        if count > 0:
            # Show sample data
            print("\nSample rows (first 5):")
            df = pd.read_sql("SELECT * FROM rl_params LIMIT 5", engine)
            print(df)
            
            # Check unique simulations
            df_sims = pd.read_sql(
                "SELECT DISTINCT simulation FROM rl_params", engine
            )
            print(f"\nUnique simulations: {df_sims['simulation'].tolist()}")
            
            # Check unique episodes
            df_episodes = pd.read_sql(
                "SELECT DISTINCT episode FROM rl_params ORDER BY episode", engine
            )
            print(f"\nUnique episodes: {df_episodes['episode'].tolist()}")
            
            # Check evaluation_mode values
            df_eval = pd.read_sql(
                "SELECT DISTINCT evaluation_mode FROM rl_params", engine
            )
            print(f"\nUnique evaluation_mode values: {df_eval['evaluation_mode'].tolist()}")
            
            # Check datetime format
            df_dt = pd.read_sql(
                "SELECT datetime FROM rl_params LIMIT 5", engine
            )
            print(f"\nSample datetime values:")
            print(df_dt)
            print(f"Datetime type: {df_dt['datetime'].dtype}")
        else:
            print("\n✗ No data in rl_params table")
else:
    print("\n✗ rl_params table does not exist")

# Check rl_grad_params too
if "rl_grad_params" in tables:
    print("\n✓ rl_grad_params table exists")
    with engine.connect() as conn:
        result = conn.execute(text("SELECT COUNT(*) FROM rl_grad_params"))
        count = result.scalar()
        print(f"Total rows in rl_grad_params: {count}")
else:
    print("\n✗ rl_grad_params table does not exist")

print("\nDone!")
