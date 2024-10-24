# Created on 24/10/2024 by Lewis Dickson 

"""
Reformatter for the API call into the standard data format for the competition 
"""

#==========================================
# User Settings 
#==========================================

data_path = r'/home/ldickson/AIForecast/Competition Data/Data/openmeteo_last10years.csv' #AQ_varaibles_2yrs.csv

#==========================================
# Imports 
#==========================================

import pandas as pd
import datetime 
import os 

#==========================================
# Functions
#==========================================

def date_format_alteration(dF, dateheader):
    for idx, date in dF[dateheader].items():
        # Convert the string to a datetime object
        date_obj = pd.to_datetime(date)
        
        # Reformat the date to the desired format
        reformatted_date = date_obj.strftime('%Y-%m-%d %H:%M:%S+00:00')
        
        # Update the DataFrame with the reformatted date
        dF.at[idx, dateheader] = reformatted_date
    
    return dF

#==========================================
# Main 
#==========================================

dF = pd.read_csv(data_path)
dF_reformatted =date_format_alteration(dF, dateheader='date')
print(dF_reformatted)
fname = os.path.basename(data_path)

dF_reformatted.to_csv(f'reformatted_{fname}', index=False)