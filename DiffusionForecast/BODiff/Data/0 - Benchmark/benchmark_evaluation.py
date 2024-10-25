import numpy as np, pandas as pd

# RMSSE calculation
def rmsse(train, test, forecast):
    forecast_mse = np.mean((test - forecast) ** 2, axis=0)
    train_mse = np.mean((np.diff(np.trim_zeros(train)) ** 2))
    return np.sqrt(forecast_mse / train_mse)

# Location 1 - Savanna Preserve
data_src = '../1 - Savanna Preserve/' # location od case study files for evaluation
forecast = pd.read_csv('1_ifs_eval.csv')
train = pd.read_csv(data_src + '1_y_train.csv',index_col=0)
test = pd.read_csv(data_src + '1_y_test.csv',index_col=0)
# rmsse (last 31 days as train for scaling)
temp_rmsse = rmsse(train.values[-31*24:,2], test.values[:,2], forecast.values[:,1])
hum_rmsse = rmsse(train.values[-31*24:,3], test.values[:,3], forecast.values[:,2])
# overall rmsse
savanna_rmsse = (temp_rmsse + hum_rmsse) / 2

# Location 2 - Clean Urban Air
data_src = '../2 - Clean Urban Air/' # location od case study files for evaluation
forecast_hum = pd.read_csv('2_ifs_eval.csv') # humidity forecast (ECMWF IFS model)
forecast_aqi = pd.read_csv('2_cams_eval.csv') # humidity forecast (ECMWF IFS model)
train = pd.read_csv(data_src + '2_y_train.csv',index_col=0)
test = pd.read_csv(data_src + '2_y_test.csv',index_col=0)
# rmsse (last 31 days as train for scaling)
hum_rmsse = rmsse(train.values[-31*24:,2], test.values[:,2], forecast_hum.values[:,1])
aqi_rmsse = rmsse(train.values[-31*24:,3], test.values[:,3], forecast_aqi.values[:,1])
# overall rmsse
urban_rmsse = (hum_rmsse + aqi_rmsse) / 2

# Location 3 - Resilient Fields
data_src = '../3 - Resilient Fields/' # location od case study files for evaluation
forecast = pd.read_csv('3_ifs_eval.csv')
train = pd.read_csv(data_src + '3_y_train.csv',index_col=0)
test = pd.read_csv(data_src + '3_y_test.csv',index_col=0)
# rmsse (last 31 days as train for scaling)
prec_rmsse = rmsse(train.values[-31*24:,2], test.values[:,2], forecast.values[:,1])
irr_rmsse = rmsse(train.values[-31*24:,3], test.values[:,3], forecast.values[:,2])
# overall rmsse
res_fields_rmsse = (temp_rmsse + hum_rmsse) / 2

# print results
print('RMSSE')
print(f"1 - Savanna Preserve: {savanna_rmsse:.3f}")
print(f"2 - Clean Urban Air: {urban_rmsse:.3f}")
print(f"3 - Resilient Fields: {res_fields_rmsse:.3f}")
print(f"Overall: {(savanna_rmsse+urban_rmsse+res_fields_rmsse)/3:.3f}")
