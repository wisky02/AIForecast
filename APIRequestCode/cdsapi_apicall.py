import cdsapi
import xarray as xr
import pandas as pd
from datetime import datetime, timedelta


latitude = -1.483719
longitude = 35.125857


c = cdsapi.Client()


end_date = datetime.today()
# start_date = end_date - timedelta(days=365 * 10)
start_date = end_date - timedelta(10)



start_date_str = start_date.strftime('%Y-%m-%d')
end_date_str = end_date.strftime('%Y-%m-%d')


c.retrieve(
    'reanalysis-era5-land',
    {
        'product_type': 'reanalysis',
        'format': 'netcdf',  # or 'grib'
        'variable': [
            '2m_temperature', 'total_precipitation', '10m_u_component_of_wind', 
            '10m_v_component_of_wind', 'surface_pressure', 'mean_sea_level_pressure',
            
        ],
        'year': [str(year) for year in range(start_date.year, end_date.year + 1)],
        'month': [f'{month:02d}' for month in range(1, 13)],
        'day': [f'{day:02d}' for day in range(1, 32)],
        'time': [
            '00:00', '01:00', '02:00', '03:00', '04:00', '05:00', 
            '06:00', '07:00', '08:00', '09:00', '10:00', '11:00', 
            '12:00', '13:00', '14:00', '15:00', '16:00', '17:00', 
            '18:00', '19:00', '20:00', '21:00', '22:00', '23:00',
        ],
        'area': [latitude, longitude, latitude, longitude],  
    },
    'output.nc'  
)


ds = xr.open_dataset('out.nc')


df = ds.to_dataframe().reset_index()

# Save the DataFrame to CSV
df.to_csv('era5_data.csv', index=False)


