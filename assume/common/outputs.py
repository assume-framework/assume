import logging
from datetime import datetime, timedelta
import os
import pandas as pd
from mango import Role
from pathlib import Path
import pandas as pd
from pathlib import Path
import yaml
from sqlalchemy import create_engine, inspect


logger = logging.getLogger(__name__)


class WriteOutput(Role):
    def __init__(self,
        simulation_id: str, 
        export_csv: bool, 
        write_orders_frequency: int,
        start_date: str,
        end_date: str,
        database_uri: str = "",
        export_csv_path: str = ""
        ):

        super().__init__()

        #store needed date
        self.simulation_id = simulation_id
        self.export_csv = export_csv
        self.write_orders_frequency = write_orders_frequency
        
        #make directory if not already present
        self.export_csv_path = export_csv_path
        self.p = Path(self.export_csv_path)
        self.p.mkdir(parents=True, exist_ok=True)
        self.db = database_uri

        #contruct all timeframe under which hourly values are written to excel and db
        start_time_dt = datetime.strptime(start_date, "%Y-%m-%d %H:%M")
        end_time_dt = datetime.strptime(end_date, "%Y-%m-%d %H:%M")
        self.timestamps = []
        current_time_dt = start_time_dt
        while current_time_dt <= end_time_dt:
            self.timestamps.append(current_time_dt.strftime("%Y-%m-%d %H:%M"))
            current_time_dt += timedelta(hours=float(self.write_orders_frequency))

        #Check id data for this simulation id is already present and delete it if so
        logger.info(
                f'deleting all data with the id {self.simulation_id} if this simulation was previously run'
            )
           
        # Loop through all Excel files in the directory
        for file_name in os.listdir(self.export_csv_path ):
                
            # Load the Excel file into a pandas dataframe
            file_path = os.path.join(self.export_csv_path , file_name)
            df = pd.read_csv(file_path)
            
            # Filter the dataframe based on the specified column and condition
            df = df[df['simulation'] != self.simulation_id]
            
            # Save the updated dataframe back to the original Excel file
            df.to_csv(file_path, index=False)
  

        #Loop throuph all database tabels
        # Get list of table names in database
        table_names = inspect(self.db.bind).get_table_names()

        # Iterate through each table
        for table_name in table_names:
            
            # Read table into Pandas DataFrame
            df = pd.read_sql_table(table_name, self.db.bind)
            print(df)
            
            # Apply filter to delete rows where a column meets a certain condition
            df = df[df['simulation'] != self.simulation_id]
            
            # Save filtered DataFrame back to table
            df.to_sql(table_name, self.db.bind, if_exists='replace', index=False)

    
    
    def setup(self):
        self.context.subscribe_message(
            self,
            self.handle_message,
            lambda content, meta: content.get("context") == "write_results",
        )
   


    def handle_message(self, content, meta):

        if content.get('type') == 'store_order_book':
            self.write_market_orders( content.get('data'), self.write_orders_frequency)

        elif content.get('type') == 'store_market_results':
            self.write_market_results(content.get('data'))

        elif content.get('type') == 'store_units':
            self.write_units_defintion(content.get('unit_type'), content.get('data'))



    def write_market_results(self, market_meta):
        df = pd.DataFrame.from_dict(market_meta)
        df['simulation']=self.simulation_id
        
        if self.export_csv:
            
            market_data_path = self.p.joinpath("market_meta.csv")
            df.to_csv(market_data_path, mode="a", header=not market_data_path.exists())

        df.to_sql("market_meta", self.db.bind, if_exists="append")



    async def write_market_orders(self, market_result):
        
        
        current_time = self.context.current_time
        if current_time == self.timestamps[0]:
            df = pd.DataFrame.from_dict(market_result)
            df["agent_id"]=df["agent_id"].astype(str)
            df['simulation']=self.simulation_id

        else:

            df=df.append(pd.DataFrame.from_dict(market_result))

        if current_time in self.timestamps: 

            if self.export_csv:
                market_data_path = self.p.joinpath("market_orders.csv")
                df.to_csv(market_data_path, mode="a", header=not market_data_path.exists())

            df.to_sql("market_orders_all", self.db.bind, if_exists="append")

            df = pd.DataFrame()





    def write_units_defintion(self, unit_type, unit_params):

        if unit_type != 'demand':

            df = pd.DataFrame([unit_params])
            df['simulation']=self.simulation_id
            df=df[['simulatio','technology','fuel_type','emission_factor','max_power','min_power','efficiency','unit_operator']]
        
            if self.export_csv:
                p = Path(self.export_csv_path)
                p.mkdir(parents=True, exist_ok=True)
                market_data_path = p.joinpath("unit_meta.csv")
                df.to_csv(market_data_path, mode="a", header=not market_data_path.exists())
            df.to_sql("unit_meta", self.db.bind, if_exists="append")

            
        else:
            df = pd.DataFrame()    
            df['type']= unit_type
            df['volume']= unit_params['volume'].max()
            df['simulation']=self.simulation_id
            #df['volume']=df["volume"].max()
            
            if self.export_csv:
                p = Path(self.export_csv_path)
                p.mkdir(parents=True, exist_ok=True)
                market_data_path = p.joinpath("demand_meta.csv")
                df.to_csv(market_data_path, mode="a", header=not market_data_path.exists())
            df.to_sql("demand_meta", self.db.bind, if_exists="append")


    async def write_dispatch_plan(self, total_power_output, unit, unit_id):
        """
        Writes the planned dispatch of the units after the market clearing to a csv and db
        In the case that we have no portfolio optimisation this equals the bids. 
        """

        df = pd.DataFrame.from_dict(total_power_output)
        df.rename(columns={df.columns[0]:'power'}, inplace=True)
        df['unit']=unit
        #sql does not liek tuples, so conversion necessary
        df['unit_id']=unit_id
        df['simulation']=self.simulation_id

        
        if self.export_csv:
            p = Path(self.export_csv_path)
            p.mkdir(parents=True, exist_ok=True)
            data_path = p.joinpath("power_plant_dispatch.csv")
            df.to_csv(data_path, mode="a", header=not data_path.exists())
        
        df.to_sql("power_plant_dispatch", self.db.bind, if_exists="append")



