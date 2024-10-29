

#csv path: probably best to give the absolute directory
path_to_csv = "../csv_data/shot_data.csv"


scale = 2

#######
#Variable dictionary:
#Start:     Starting coordinate for the variable. Best to choose either a known
#           decent coordinate or somehwere near the centre of the parameter
#           space.
#Scale:     Parameter used for normalising the variables. This value should be
#           5-20% of the size of the parameter space (larger for more
#           exploration)
#Bounds:    Boundaries of the parameter space.
#######



# var_names = ['x1', 'x2']
# start_vals = [0 , 0]
# var_scales = [2, 2]
# var_bounds = [[-5,5], [-5,5]]
# var_dict = create_var_dict(var_names, start_vals, var_scales, var_bounds)

#Example dictionaries:
var_dict = {'name':['x1'],
                 'start':[0 ],
                 'scale':[scale],
                 'bounds': [[-5,5]]
                 }
var_dict = {'name':['focus', 'o2'],
                'start':[0., 7000],
                'scale':[0.05, 3000],
                'bounds': [[-0.2,0.2], [1000,15000]]
               }

"""

var_dict = {'name':['x1', 'x2', 'x3'],
                'start':[-3 , -3,-3],
                'scale':[scale, scale,scale],
                'bounds': [[-5,5], [-5,5],[-5,5]]
               }

var_dict = {'name':['x1', 'x2', 'x3','x4'],
                'start':[0 , 0,0,0],
                'scale':[scale, scale,scale,scale],
                'bounds': [[-5,5], [-5,5], [-5,5],[-5,5]]
               }

var_dict = {'name':['x1', 'x2', 'x3','x4','x5'],
                'start':[0 , 0,0,0,0],
                'scale':[scale, scale,scale,scale,scale],
                'bounds': [[-5,5], [-5,5], [-5,5],[-5,5],[-5,5]]
               }

var_dict = {'name':['x1', 'x2', 'x3','x4','x5', 'x6'],
                'start':[-3 , -3,-3,-3,-3,-3],
                'scale':[scale, scale,scale,scale,scale,scale],
                'bounds': [[-5,5], [-5,5], [-5,5],[-5,5],[-5,5],[-5,5]]
               }

"""

NDIMS = len(var_dict['name'])

#Number of random samples before beginning optimisation
N_rand = NDIMS
