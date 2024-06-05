import os
import numpy as np
# from numpy import pi, sqrt


param_names = ["u, -r", "a", "b", "phi", "N", "dt"]
param_title = ["u, -r", "\\alpha", "\\beta", "{\\bar \\varphi}", "N", "\\Delta t"]

L = 10.

def filename_from_param(param):
    return ''.join(param_names[i] + '=' + str(param[i]) + '_' for i in range(len(param_names)))[:-1]

def title_from_param(param):
    return "$" + ''.join(param_title[i] + '=' + "%.3f" % param[i] + '\\quad' for i in range(len(param_names))) + "$"

def get_all_filenames_in_folder(folder_path):
    """
    Returns a list of all filenames of all files in a folder.
    """
    filenames = []
    for filename in os.listdir(folder_path):
        if os.path.isfile(os.path.join(folder_path, filename)):
            if filename[:5]=="noise":
                continue
            filenames.append(filename)
    return filenames

def param_dict_from_filename(s):
    """
    Extracts parameters from a string in the format 'value1=xxx_value2=yyy...' and
    returns a dictionary in the format {'value1': xxx, 'value2': yyy, ...}.
    """
    params = {}
    for param in s.split('_'):
        key, value = param.split('=')
        params[key] = float(value)
    return params

def param_from_dict(param_dict):
    return [param_dict[key] for key in param_names]

def param_from_filename(filename):
    return param_from_dict(param_dict_from_filename(filename))

def load_file(folder, filename):
    file = folder+filename+'.txt'
    noisefile = folder+"noise:"+filename+".txt"
    param = param_from_filename(filename)

    u, a, b, phi, N, dt = param
    N = int(N)
    param = u, a, b, phi, N, dt

    phit = np.loadtxt(file)
    xit = np.loadtxt(noisefile)
    frames = len(phit)
    phit = phit.reshape((frames, 2, N))
    xit = xit.reshape((frames-1, 2, N)) 
    # xit = None
    return phit, xit, param
