from scipy.stats import multivariate_normal
import numpy as np

class Ndim_multinorm:
    #Makes an N dimensional multivariate normal distribution (single peak)
    def __init__(self, Ndim, mean =None,cov_mat = None ):
        if list(mean):
            mean = mean
            cov_mat = (cov_mat + cov_mat.T)/2
            cov_mat = np.matmul(cov_mat.T, cov_mat)
        else:
            #generate position of maximum
            mean = np.random.normal(size = Ndim)

            #generate random covariance matrix
            cov_mat=np.random.normal(size = (Ndim,Ndim))
            cov_mat = (cov_mat + cov_mat.T)/2
            cov_mat = np.matmul(cov_mat.T, cov_mat)
            #create function
        #make cov_mat positive semidefinte and symmetric

        #create function
        self.rv = multivariate_normal(mean =mean, cov = cov_mat)

        #normalisation_coeff:
        self.max = self.rv.pdf(mean)


        self.mean = mean
        self.cov_mat = cov_mat

    def ask(self, coord):
        return self.rv.pdf(coord)/self.max

    def params(self):
        return self.mean, self.cov_mat




class Ndim_multinorm_multipeak:
    #combines a list of multivariate normal functions

    def __init__(self,means,cov_mats, amplitudes, multinorms = None, noise = 0  , normalise = True):


        self.multinorms = []
        print(len(means))
        for i, mean in enumerate(means):
            print(means[i])
            print(cov_mats[i])
            f = multivariate_normal(mean =means[i], cov = cov_mats[i])

            self.multinorms.append(f)

        self.amplitudes = amplitudes
        self.noise = noise
        self.max_val = 1


        #normalise
        if normalise:
            max_coord = means[np.argmax(amplitudes)]
            self.max_val = self.ask(max_coord, noise = False)


    def ask(self, coord, noise=True):
        result = 0
        for i, rv in enumerate(self.multinorms):
            result+= rv.pdf(coord) *self.amplitudes[i]
        if noise:
            result+=np.random.normal(0,0.01)*self.noise
        result =result/self.max_val
        return result
