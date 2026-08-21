
import numpy as np
from lmfit import Parameters
from scipy.special import factorial
import re

def load_data(filepath):
    """
    Loads a pre-merged, whitespace-separated MCD data matrix.
    """

    with open(filepath, 'r') as f:
        header_line = f.readline().strip()
        headers = re.split(r'\s{2,}', header_line)
        
    data = np.loadtxt(filepath, skiprows=1)
    
    x_cm = data[:, 0][::-1]
    
    final_data = []
    for i in range(1, data.shape[1]):
        final_data.append(data[:, i][::-1])
        
    labels = headers[1:] 
    
    return x_cm, final_data, labels

def read_params(file):
    """
    Reads in bands, amps, and constraints from input file. Format is\n
        bands\n
        7  10000  400    600    2.5   0  1 1 1 1 \n       
        1  11000  1400     0    0       0 1 1 1 1     \n
        end\n

        amps\n
        205       63.9    73.3   80.7   59.2\n
        -264.8    -72.0  -109.6 -157.9 -123.9\n
        end\n

        constraints\n
        0 -= 1\n
        2 -= 3\n
        end\n


    Where each row in bands contains\n \n
    nbands center width split Sfac 0/1 0/1 0/1 0/1 0/1\n
     \n
    with the following 0/1 stands for vary = (False/True) for each of the parameters in order.\n
    nbands greater than 1 represents a vibronic progression that then uses the split for the spacing between peaks and Sfac as the Huang-Rhys factor.\n
    For normal guassian functions set 'nbands' = 1, split = 0, Sfac = 0.\n
    In amps each line contains the ordered amplitudes for the specific band for each dataset. Positive amps are constrainted to stay positive and negative amps negative.\n
    There must be the same number of amps for all bands, if a band is not needed use 0 as it will not float.\n
    constraints is formatted as\n
    band1 operator band2\n
    where operator is -= or +=

    """
    with open(file,'r') as f:
        bands = []
        amps = []
        constraints = []
        band_copy = False
        amps_copy = False
        const_copy = False
        for line in f:
            if 'end' in line.strip():
                band_copy = False
                amps_copy = False
                const_copy = False
            if '#' in line.strip():
                continue
            if 'bands' in line.strip():
                band_copy = True
                continue
            if band_copy == True:
                line = line.strip(',').split()
                bands.append(np.genfromtxt(line))
                continue
            if 'amps' in line.strip():
                amps_copy = True
                continue
            if amps_copy == True:
                line = line.strip(',').split()
                amps.append(np.genfromtxt(line))
                continue
            if 'constraints' in line.strip():
                const_copy = True
                continue
            if const_copy == True:
                line = line.strip(',').split()
                constraints.append(line)
                continue
        return bands,amps,constraints

def parameter_gen(bands,amps,constraints = None):
    """ Generates the parameters used for fitting. Reads in the bands and amps lists.\n
            The format is variable_{band}_{spectra}\n
    The number of amplitudes (ie the number of spectra) for each band is the outer loop and the number of bands in each spectra is the inner loop. \n
    check_sign is used to enforce negative amplitudes stay negative, and positive stay positive \n
    The amp, center, and fwhm have the values and vary=True/False (1/0) set by the bands list. Amplitudes will always vary unless set to 0.
    If vibronic progression is used, nbands = number of bands in vibronic progression, split is the spacing (vibrational energy), and Sfac is the Huang-Rhys factor \n
    If there are more than one spectra (amps), the center and fwhm to the same value as the first given. Split and Sfactor stay constant for each band.
    If constraints are added to keep band amplitudes opposite signed and equal, the final loop enforces this via setting an expression for the amplitudes
    """
    fit_params = Parameters()
    for i in range(len(amps[0])):
        for j in range(len(bands)):
            min_max = check_sign(amps[j][i])       
            fit_params.add(f'amp_{j}_{i}', value = amps[j][i], min = min_max[0], max = min_max[1], vary = min_max[2])
            fit_params.add(f'center_{j}_{i}', value = bands[j][1], min = None, max = None, vary = bands[j][6])
            fit_params.add(f'fwhm_{j}_{i}', value = bands[j][2], min = None, max = None, vary = bands[j][7])
            if bands[j][0] > 1:
                fit_params.add(f'nbands_{j}_{i}', value = bands[j][0], vary = bands[j][5] )
                fit_params.add(f'split_{j}_{i}', value = bands[j][3], min = 0, max = bands[j][3]*2, vary = bands[j][8])
                fit_params.add(f'Sfac_{j}_{i}', value = bands[j][4], min = 0, max = None, vary = bands[j][9])
            if i > 0:
                fit_params[f'center_{j}_{i}'].expr = f'center_{j}_0'
                fit_params[f'fwhm_{j}_{i}'].expr = f'fwhm_{j}_0'
                if bands[j][0] > 1:
                    fit_params[f'split_{j}_{i}'].expr = f'split_{j}_0'
                    fit_params[f'Sfac_{j}_{i}'].expr = f'Sfac_0_0'
            if (bands[j][0] > 1) and (j > 0):
                fit_params[f'Sfac_{j}_{i}'].expr = f'Sfac_0_0'
    for j in range(len(constraints)):
        for i in range(len(amps[0])):
            fit_params[f'amp_{int(constraints[j][0])}_{i}'].expr = constraints[j][1][0] + f'amp_{int(constraints[j][2])}_{i}'
    return fit_params

def check_sign(amp):
    if amp > 0:
        
        min_max = [0,None,True]
    elif amp == 0:
        min_max = [None,None,False]
    else:
        min_max = [None,0,True]
    return min_max

def objective(params,x,bands,data):
    """Reads in params generated from parameters_gen(), the x from the range being fit, the total number of bands to fit, and the data as a list. \n
    Uses spec_mode() to calculate the individual residuals for each spectra and return as a multidimensional array that is then flattened for the minimizer:
    """
    resid = np.zeros((len(data),len(x)))
    for i in range(len(data)):
        resid[i] = data[i] - spec_model(bands,params,x,i)
    return resid.flatten()

def gaussian(x,center,amp,fwhm):
    """Gaussian lineshape calculator that changes the given fwhm to sigma"""
    sigma = fwhm / 2.35482
    return amp * np.exp(-(x - center)**2 / (2 * sigma**2))

def vibronic(bandcount,params,n,m,x):
    """Generates vibronic progression using the Huang-Rhys factor and Poisson distribution with indiviual Guassian lineshapes. \n
    Bandcount = len(band)\n 
    params comes of parameter_gen()\n 
    n (band) and m (spectra) are the indices from outer loop passed into function
    """
    amp = params[f'amp_{n}_{m}']
    cen = params[f'center_{n}_{m}']
    fwhm = params[f'fwhm_{n}_{m}']
    hr = params[f'Sfac_{n}_{m}']
    split = params[f'split_{n}_{m}']
    i_vals = np.arange(1, bandcount +1)
    vib_amps = (amp * (hr** i_vals) / factorial(i_vals)) / bandcount
    centers = cen + split * np.arange(bandcount)
    sigma = fwhm / 2.35482
    exponent = -((x[np.newaxis, :] - centers[:, np.newaxis]) ** 2) / (2 * sigma ** 2)
    all_bands = vib_amps[:, np.newaxis] * np.exp(exponent)

    return np.sum(all_bands, axis=0)

def spec_model(bands,params,x,i):
    """
    Generates the model from the parameters for each individual spectra.\n
    bands = bands list\n
    params = parameters from parameter_gen()\n
    x = x data range\n
    i = the iteration of the outer loop, which corresponds to the individual spectra being fit\n
    This builds the Gaussian and vibronic bands from all parameters and sums them to be used in the outer loop residual calculation
    """
    model = np.zeros(len(x))
    for j in range(len(bands)):
        if bands[j][0] == 1:
            model += gaussian(x,params[f'center_{j}_{i}'],params[f'amp_{j}_{i}'],params[f'fwhm_{j}_{i}'])
        if bands[j][0] > 1:
            model += vibronic(int(params[f'nbands_{j}_{0}'].value),params,j,i,x)
    return model


def spec_bands(bands,params,x,i):
    """
    Used for plotting, this generates the individual bands for each spectra.\n
    bands = list of band information\n
    params = parameters from parameter_gen() for initial, or in the case of final fits, out.params from the minimization\n
    x = x range the spectra are fitted over\n
    i = the individual spectra, passed from the loop outside the function
    """
    allbands = []
    for j in range(len(bands)):
        if bands[j][0] == 1:
            allbands.append(gaussian(x,params[f'center_{j}_{i}'],params[f'amp_{j}_{i}'],params[f'fwhm_{j}_{i}']))
        if bands[j][0] > 1:
            allbands.append(vibronic(int(params[f'nbands_{j}_{0}'].value),params,j,i,x))
    return allbands        

def output_lists(out,bands,amps):
    outbands = [[0]*10 for i in range(len(bands))]
    outamps = [[0] * len(amps[0]) for i in range(len(amps))]
    for i in range(len(bands)):
        outbands[i][0] = int(bands[i][0])
        outbands[i][1] = np.around(out.params[f'center_{i}_{0}'].value,decimals=1)
        outbands[i][2] = np.around(out.params[f'fwhm_{i}_{0}'].value,decimals=1)
        for j in range(5,10):
            outbands[i][j] = int(bands[i][j])

        if bands[i][0] > 1:
            outbands[i][3] = np.around(out.params[f'split_{i}_{0}'].value,decimals=1)
            outbands[i][4] = np.around(out.params[f'Sfac_{i}_{0}'].value,decimals=3)
    for i in range(len(amps[0])):
        for j in range(len(amps)):
            outamps[j][i] = np.around(out.params[f'amp_{j}_{i}'].value,decimals=1)
    return outbands, outamps