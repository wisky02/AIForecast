# Created on 22/10/2024 by Lewis Dickson  

#================================================================
# User Settings
#================================================================

# %%
h_lim = 168 #hours forecasting window
input_folder =r'./input_data/'
api_key = 'nixak-2LCbrvDu8uh6uONR7nnoU8L8i0B0ZR2UOpdNfiBnZs4S5vdFe73JXyDGz7r15mABHL5H7Xvmva2oCbXd' # your api 

# --- Analysis Options --- # 
fine_tune = True 
set_finetune_steps = 500


#================================================================
# Imports
#================================================================

import pandas as pd
from nixtla import NixtlaClient
from utilsforecast.losses import mae, mse
from utilsforecast.evaluation import evaluate
import matplotlib.pyplot as plt

#================================================================
# Main -Forecast
#================================================================

# Launch API client
nixtla_client = NixtlaClient(
    api_key = api_key
)

# Validate API Key
if nixtla_client.validate_api_key():
    print('API Key Valid')
else:
    raise ValueError("API Key is not Valuid")

# Extract GT training data 
df = pd.read_csv(rf'{input_folder}/1_y_train.csv')

# Temperature inference
timegpt_fcst_df_temperature = nixtla_client.forecast(df=df, h=168, freq='h', time_col='date', target_col='temperature_2m', model='timegpt-1-long-horizon')

# Humidity inference 
timegpt_fcst_df_humidity = nixtla_client.forecast(df=df, h=168, freq='h', time_col='date', target_col='relative_humidity_2m', model='timegpt-1-long-horizon')

# Load the test data 
df_test = pd.read_csv(rf'{input_folder}/1_y_test.csv')

# Convert to datetime format (standard output from timeGPT)
df_test['date'] = pd.to_datetime(df_test['date'], utc=True)

# Combine X and y for easier analysis, non-matching dates are dropped
dF_valid = pd.merge(df_test, timegpt_fcst_df_temperature, on=['date'], how='inner')
dF_valid = pd.merge(dF_valid, timegpt_fcst_df_humidity, on=['date'], how='inner')

plt.plot(dF_valid['TimeGPT_x'], 'r', label ='Forecasted Temperature')
plt.plot(dF_valid['temperature_2m'], 'b', label ='Actual Temperature')
plt.legend()
plt.title('Forecasted Temperature vs Actual Temperature')
plt.show()

plt.plot(dF_valid['TimeGPT_y'], 'r', label ='Forecasted Humidity')
plt.plot(dF_valid['relative_humidity_2m'], 'b', label ='Actual Humidity')
plt.legend()
plt.title('Forecasted Humidity vs Actual Humidity')
plt.show()

if False: # plot of raw difference for each day
    dF_valid['Diff'] = dF_valid['temperature_2m'] - dF_valid['TimeGPT_x']
    plt.plot(dF_valid['Diff'])
    plt.show

#================================================================
# Main -Fine tuning
#================================================================

if fine_tune:
    # Perfrom fine tuning  
    """
    control the number of steps with finetune_steps
    Note: may need different values for the temp vs humidity
    """

    timegpt_fcst_finetune_df_temperature = nixtla_client.forecast(
        df=df, h=h_lim, finetune_steps=set_finetune_steps,
        time_col='date', target_col='temperature_2m',
        model='timegpt-1-long-horizon'
    )

    timegpt_fcst_finetune_df_humidity = nixtla_client.forecast(
        df=df, h=h_lim, finetune_steps=set_finetune_steps,
        time_col='date', target_col='relative_humidity_2m',
        model='timegpt-1-long-horizon'
    )

    # Load the test data 
    df_test = pd.read_csv(rf'{input_folder}/1_y_test.csv')


    # Convert to datetime format
    df_test['date'] = pd.to_datetime(df_test['date'], utc=True)

    # Combine X and y for easier analysis
    dF_valid = pd.merge(df_test, timegpt_fcst_finetune_df_temperature, on=['date'])
    dF_valid = pd.merge(dF_valid, timegpt_fcst_finetune_df_humidity, on=['date'])
    print(dF_valid)

    plt.plot(dF_valid['TimeGPT_x'], 'r', label ='Forecasted Temperature')
    plt.plot(dF_valid['temperature_2m'], 'b', label ='Actual Temperature')
    plt.legend()
    plt.title('Fine-Tuned Forecasted Temperature vs Actual Temperature')
    plt.show()

    plt.plot(dF_valid['TimeGPT_y'], 'r', label ='Forecasted Humidity')
    plt.plot(dF_valid['relative_humidity_2m'], 'b', label ='Actual Humidity')
    plt.legend()
    plt.title('Fine-Tuned Forecasted Humidity vs Actual Humidity')
    plt.show()