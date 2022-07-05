import numpy as np
import glm

def convert2glm(x):
    return glm.quat(x[0],x[1],x[2],x[3])

def convert2array(x):
    return np.array([x[3],x[0],x[1],x[2]])

def convertfromvec2glm(x):
    return glm.quat(0, x[0],x[1],x[2])

def convertfromglm2vec(x):
    return np.array([x[0],x[1],x[2]])
