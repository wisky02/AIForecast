# timeGPT fine tuning

#%%

h_lim = 168 #hours forecasting window


import pandas as pd
from nixtla import NixtlaClient
from utilsforecast.losses import mae, mse
from utilsforecast.evaluation import evaluate
api_key = 'nixak-2LCbrvDu8uh6uONR7nnoU8L8i0B0ZR2UOpdNfiBnZs4S5vdFe73JXyDGz7r15mABHL5H7Xvmva2oCbXd'

nixtla_client = NixtlaClient(
    api_key = api_key
)

# print(nixtla_client.validate_api_key())
import pandas as pd
df = pd.read_csv(r'C:\Users\Lewis\Documents\MLProjects\AIForecast\Competition Data\AugmentedData\1 - Savanna Preserve/1_y_train.csv')



timegpt_fcst_finetune_df_temperature = nixtla_client.forecast(
    df=df, h=h_lim, finetune_steps=500,
    time_col='date', target_col='temperature_2m',
    model='timegpt-1-long-horizon'
)

timegpt_fcst_finetune_df_humidity = nixtla_client.forecast(
    df=df, h=h_lim, finetune_steps=500,
    time_col='date', target_col='relative_humidity_2m',
    model='timegpt-1-long-horizon'
)

# %%

# Load the test data 
df_test = pd.read_csv(r'C:\Users\Lewis\Documents\MLProjects\AIForecast\Competition Data\AugmentedData\1 - Savanna Preserve/1_y_test.csv')

# Convert to datetime format
df_test['date'] = pd.to_datetime(df_test['date'], utc=True)

# Find the accuracy of the forecast from the train data vs the test data

df_test = df_test.head(h_lim)
# diff_list = timegpt_fcst_df['TimeGPT'] - df_test['temperature_2m']

# Combine X and y for easier analysis
dF_comb = pd.merge(df_test, timegpt_fcst_finetune_df_temperature, on=['date'])
dF_comb = pd.merge(dF_comb, timegpt_fcst_finetune_df_humidity, on=['date'])
print(dF_comb)

import matplotlib.pyplot as plt
# plt.plot(dF_comb['TimeGPT_x'], 'r')
# plt.plot(dF_comb['temperature_2m'], 'b')
# plt.show()

plt.plot(dF_comb['TimeGPT_y'], 'r')
plt.plot(dF_comb['relative_humidity_2m'], 'b')
plt.show()

print(f'{dF_comb}')
dF_comb['Diff'] = dF_comb['temperature_2m'] - dF_comb['TimeGPT_x']
plt.plot(dF_comb['Diff'])
plt.show

# %%
