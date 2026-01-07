#!/usr/bin/env python3
"""Monitor database writes in real-time"""

import time
import pandas as pd
from sqlalchemy import create_engine

db_uri = "postgresql://assume:assume@localhost:5432/assume"
engine = create_engine(db_uri)

simulation_id = "2_nodes_learning_single_diesel"

print(f"Monitoring database for simulation: {simulation_id}")
print("Press Ctrl+C to stop\n")

last_count = 0
try:
    while True:
        query = f"""
        SELECT COUNT(*) as count
        FROM rl_params 
        WHERE simulation = '{simulation_id}'
        """
        
        df = pd.read_sql(query, engine)
        current_count = df['count'][0]
        
        if current_count != last_count:
            print(f"[{time.strftime('%H:%M:%S')}] rl_params rows: {current_count} (+{current_count - last_count})")
            last_count = current_count
        
        time.sleep(2)
except KeyboardInterrupt:
    print("\n\nStopped monitoring")
    print(f"Final count: {last_count} rows")
