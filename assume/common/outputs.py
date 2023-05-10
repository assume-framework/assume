import logging
from datetime import datetime, timedelta
import os
import pandas as pd
import assume
from mango import Role
from pathlib import Path
import pandas as pd
from pathlib import Path
import yaml
from sqlalchemy import inspect
#from mango.agent import Agent, scheduler
import time
import asyncio




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
        self.start_time_dt = datetime.strptime(start_date, "%Y-%m-%d %H:%M")
        self.end_time_dt = datetime.strptime(end_date, "%Y-%m-%d %H:%M")
        self.delay=(self.end_time_dt- self.start_time_dt).total_seconds()

        #initalizes dfs for storing and writing asynchron
        self.df_orders = pd.DataFrame()
        self.df_dispatch = pd.DataFrame()
        

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
        '''
        h=self.write_orders_frequency*3600
        

        while h < self.delay:

            self.context.schedule_timestamp_task(coroutine=self.store_market_orders(),
                                     timestamp=(self.start_time_dt + timedelta(seconds=h)).timestamp())
            
            self.context.schedule_timestamp_task(coroutine=self.store_dispatch_plan(),
                            timestamp=(self.start_time_dt + timedelta(seconds=h)).timestamp())
            h = h + self.write_orders_frequency*3600

        self.context.schedule_timestamp_task(coroutine=self.store_market_orders(),
                            timestamp=self.end_time_dt.timestamp())
        self.context.schedule_timestamp_task(coroutine=self.store_dispatch_plan(),
                            timestamp=self.end_time_dt.timestamp())
        '''


   


    def handle_message(self, content, meta):

        if content.get('type') == 'store_order_book':
            self.write_market_orders(content.get('data'))

        elif content.get('type') == 'store_market_results':
            self.write_market_results(content.get('data'))

        elif content.get('type') == 'store_units':
            self.write_units_defintion(content.get('unit_type'), content.get('data'))

        elif content.get('type') == 'store_dispatch':
            self.write_dispatch_plan(content.get('unit'), content.get('unit_id'), content.get('capacity'), content.get('timestamp'))



    def write_market_results(self, market_meta):
        df = pd.DataFrame.from_dict(market_meta)
        df['simulation']=self.simulation_id
        
        if self.export_csv:
            
            market_data_path = self.p.joinpath("market_meta.csv")
            df.to_csv(market_data_path, mode="a", header=not market_data_path.exists())

        df.to_sql("market_meta", self.db.bind, if_exists="append")


    def store_market_orders(self):
        print('we should store orders now)')
        if self.export_csv:
            market_data_path = self.p.joinpath("market_orders.csv")
            self.df_orders.to_csv(market_data_path, mode="a", header=not market_data_path.exists())

        self.df_orders.to_sql("market_orders_all", self.db.bind, if_exists="append")

        self.df_orders = pd.DataFrame()


    def write_market_orders(self, market_result):
       
        df = pd.DataFrame.from_dict(market_result)
        df['simulation']=self.simulation_id
        df=df.astype(str)

        self.df_orders=self.df_orders._append(df)


    def write_units_defintion(self, unit_type, unit_params):
        
        if unit_type=='power_plant':
            
            df = pd.DataFrame([unit_params])
            df['simulation']=self.simulation_id
            df=df[['simulation','technology','fuel_type','emission_factor','max_power','min_power','efficiency','unit_operator']]
        
            if self.export_csv:
                p = Path(self.export_csv_path)
                p.mkdir(parents=True, exist_ok=True)
                market_data_path = p.joinpath("unit_meta.csv")
                df.to_csv(market_data_path, mode="a", header=not market_data_path.exists())

            df.to_sql("unit_meta", self.db.bind, if_exists="append")

            
        elif unit_type=='demand':
            
            df = pd.DataFrame.from_dict(unit_params)    
            df['type']= unit_type
            df.reset_index(inplace=True)
            df=df.rename(columns={'level_0':'', 'index':'Timestamp'})
            #sql does not like Timestamp or other types of values
            df=df.astype(str)
            df['volume']= unit_params['volume'].max()
            df['simulation']=self.simulation_id
            #df['volume']=df["volume"].max()
            
            if self.export_csv:
                p = Path(self.export_csv_path)
                p.mkdir(parents=True, exist_ok=True)
                market_data_path = p.joinpath("demand_meta.csv")
                df.to_csv(market_data_path, mode="a", header=not market_data_path.exists())
            df.to_sql("demand_meta", self.db.bind, if_exists="append")

        else:
            logger.info(
                f'added unit not stored in db, since type {unit_type} is not yet written in outputs.py'
            )
        

    def store_dispatch_plan(self):
        print('we should store dispatch now)')
        if self.export_csv:
            p = Path(self.export_csv_path)
            p.mkdir(parents=True, exist_ok=True)
            data_path = p.joinpath("power_plant_dispatch.csv")
            self.df_dispatch.to_csv(data_path, mode="a", header=not data_path.exists())
        
        self.df_dispatch.to_sql("power_plant_dispatch", self.db.bind, if_exists="append")

    def write_dispatch_plan(self, unit, unit_id, total_power_output, current_time):
        """
        Writes the planned dispatch of the units after the market clearing to a csv and db
        In the case that we have no portfolio optimisation this equals the bids. 
        """
       
        df = pd.DataFrame.from_dict(total_power_output)
        df.rename(columns={df.columns[0]:'power'}, inplace=True)
        df['unit']=unit
        #sql does not liek tuples, so conversion necessary
        df['unit_id']=unit_id
        df['timestamp']=current_time
        df['simulation']=self.simulation_id

        self.df_dispatch=self.df_dispatch._append(df)



