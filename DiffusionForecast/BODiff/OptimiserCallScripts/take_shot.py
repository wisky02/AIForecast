import numpy as np
from BlackBox import Ndim_multinorm, Ndim_multinorm_multipeak
import pandas as pd
import os
import time
from var_dict import *


if __name__ == '__main__':
    print('Iniitialising multi peak Blackbox function')
    # means = [np.zeros(len(var_dict['name'])),
    #             3*np.ones(len(var_dict['name']))]
    # print('means', means)
    # cov_mats = [2*np.ones(len(var_dict['name']))
    #             , 2*np.ones(len(var_dict['name']))]
    means = [[-0.0, 6000], [0.15,12000]]
    cov_mats = [[0.01, 1e6], [0.01, 1e6]]
    amplitudes = [0.5,1]
    amplitudes = [7e5,1e6]
    noise = 1e0

    BB = Ndim_multinorm_multipeak(means, cov_mats, amplitudes, noise = noise, normalise=True)

    #assert os.path.exists(path_to_csv)

    while(True):
        #input("Press Enter to continue...")
        df = pd.read_csv(path_to_csv)
        next_coords = df[var_dict['name']].iloc[-1].values

        #take shot
        result = BB.ask(next_coords)

        #Error bodge
        if noise == 0:
            result_err = result*0.1
        else:
            result_err = noise *0.01
        result_err =0.01
        print('Result = ', result, '+-',result_err )

        #Attach result of shot
        df['yVal'].iat[-1]  = result
        df['yVal_err'].iat[-1] = result_err
        _ = var_dict['name']
        columns = _.copy()
        columns.insert(0, 'shot_num')
        columns.append('yVal')
        columns.append('yVal_err')
        columns.append('xi')

        df = df[columns]
        print(df)

        #Save
        df.to_csv(path_to_csv, index = False)

        #watch file for changes
        orig_time = os.path.getmtime(path_to_csv)
        while(orig_time == os.path.getmtime(path_to_csv)):
            time.sleep(0.5)
            #print(os.path.getmtime(path_to_csv))
            #print(orig_time)
