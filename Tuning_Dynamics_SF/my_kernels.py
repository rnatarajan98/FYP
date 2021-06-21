# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 15:15:12 2020

@author: John
"""

import numpy as np
import quantities as pq
import matplotlib.pyplot as plt

def create_gauss_ft(A=1, a=0.62*pq.deg,
                    dx=0.*pq.deg, dy=0.*pq.deg,xscale=1,yscale=1,theta=0):
    """
    Create Fourier transformed
    Gaussian function closure.

    Parameters
    ----------
    A : float
        peak value
    a : float/quantity scalar
        Width
    dx: float/quantity scalar
        shift in x-direction
    dy: float/quantity scalar
        shift in y-direction

    Returns
    -------
    out : function
        Evaluate function
    """
    #print('open evaluate called')
    
	#raise NameError

    def evaluate(kx, ky):
        """
        Evaluates the Fourier transform of gauss function

        Parameters
        ----------
        kx : float/quantity scalar
        ky : float/quantity scalar

        Returns
        -------
        out : ndarray
            Calculated values
        """
        #print("Custom spatial gaussian created. Parameters: "
        #      "dx=", dx, " dy=", dy,
        #      "xscale=", xscale, "yscale=", yscale,
        #      "theta=", theta)
        kxr = (kx*np.cos(theta)-ky*np.sin(theta))*xscale
        kyr = (kx*np.sin(theta)+ky*np.cos(theta))*yscale
        k2 = kxr**2 + kyr**2
        #print('evaluate called',A * np.exp(-a**2 * k2/4.) * np.exp(-1j * (kx*dx + ky*dy)))


        return A *1/(xscale*yscale) * np.exp(-a**2 * k2/4.) * np.exp(-1j * (kx*dx + ky*dy))



    return evaluate

def create_temp_gauss_ft(A=0.1, a=1*pq.ms,
                    delay=0*pq.ms):
    """
    Create Fourier transformed
    Gaussian function closure.
    Parameters
    ----------
    A : float
        peak value
    a : float/quantity scalar
        Width

    Returns
    -------
    out : function
        Evaluate function
    """
    def evaluate(w):
        """
        Evaluates the Fourier transform of gauss function
        Parameters
        ----------
        kx : float/quantity scalar
        ky : float/quantity scalar
        Returns
        -------
        out : ndarray
            Calculated values
        """
        #w = np.array(w, dtype=float)
        """
        data = A * np.exp(-a**2 * w**2/4.) * np.exp(-1j * (w*dx + w*dy))
        print(np.squeeze(data).shape)
        plt.plot(np.squeeze(data))
        """
        #plt.figure(2)
        #plt.plot(np.array(A * np.exp(-a**2 * w**2/4.) * np.exp(-1j * (w*-delay))))
        ##plt.ylabel('some numbers')
        #plt.show()
        #print('w',w.units)
        #print("Custom temporal gaussian created. Parameters: A=", A, " a=", a,"delay=",delay)
        return A * np.exp(-a**2 * w**2/4.) * np.exp(-1j * (w*-delay))
    return evaluate