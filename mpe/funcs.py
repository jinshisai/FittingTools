# coding: utf-8

# modules
import numpy as np

# gaussian
def gauss1d(x,amp,mean,sig):
    return amp * np.exp(-(x-mean)*(x-mean)/(2.0*sig*sig))

# For debug
def gaussian2d(x, y, A, mx, my, sigx, sigy, peak=True):
    '''
    Generate normalized 2D Gaussian

    Parameters
    ----------
     x: x value (coordinate)
     y: y value
     A: Amplitude. Not a peak value, but the integrated value.
     mx, my: mean values
     sigx, sigy: standard deviations
    '''
    coeff = A if peak else A/(2.0*np.pi*sigx*sigy)
    expx = np.exp(-(x-mx)*(x-mx)/(2.0*sigx*sigx))
    expy = np.exp(-(y-my)*(y-my)/(2.0*sigy*sigy))
    return coeff*expx*expy