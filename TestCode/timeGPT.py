# Created on 22/10/2024 by Lewis Dickson  

#================================================================
# User Settings
#================================================================

# %%
h_lim = 168 #hours forecasting window
input_folder =r'./input_data/'
api_key = 'nixak-2LCbrvDu8uh6uONR7nnoU8L8i0B0ZR2UOpdNfiBnZs4S5vdFe73JXyDGz7r15mABHL5H7Xvmva2oCbXd' # your api 

#================================================================
# Imports
#================================================================

from nixtla import NixtlaClient
import pandas as pd
import matplotlib.pyplot as plt

#================================================================
# Main -Forecast
#================================================================

# Launch API client
nixtla_client = NixtlaClient(
    api_key = api_key
)

# print(nixtla_client.validate_api_key())   
df = pd.read_csv(rf'{input_folder}/1_y_train.csv')

# Temperature
timegpt_fcst_df_temperature = nixtla_client.forecast(df=df, h=168, freq='h', time_col='date', target_col='temperature_2m', model='timegpt-1-long-horizon')

# Humidity
timegpt_fcst_df_humidity = nixtla_client.forecast(df=df, h=168, freq='h', time_col='date', target_col='relative_humidity_2m', model='timegpt-1-long-horizon')

# Load the test data 
df_test = pd.read_csv(rf'{input_folder}/1_y_test.csv')

# Convert to datetime format
df_test['date'] = pd.to_datetime(df_test['date'], utc=True)

# Combine X and y for easier analysis
dF_valid = pd.merge(df_test, timegpt_fcst_df_temperature, on=['date'], how='inner')
dF_valid = pd.merge(dF_valid, timegpt_fcst_df_humidity, on=['date'], how='inner')
print(dF_valid)

plt.plot(dF_valid['TimeGPT_x'], 'r')
plt.plot(dF_valid['temperature_2m'], 'b')
plt.show()
print(f'{dF_valid}')

dF_valid['Diff'] = dF_valid['temperature_2m'] - dF_valid['TimeGPT_x']
plt.plot(dF_valid['Diff'])
plt.show

#================================================================
# Main -Fine tuning
#================================================================