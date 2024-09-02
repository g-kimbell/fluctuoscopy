"""
This module provides functions to calculate the fluctuation conductivity of a 2-dimensional
superconductor in the absence of magnetic field using the FSCOPE C++ program. The functions are 
multi-threaded and wrapped in a way that allows easier use of e.g. least squares fitting algorithms.

We also include the effects of weak localisation and a varying elastic scattering time with
backgate to simplify the common process of calculating sheet resistance in a field-effect
device.

The module requires the FSCOPE program to be compiled and placed in the same directory as this file.

The most important function is 'fscope_delta_wrapped', which calculates the resistance and all
contributions to conductivity of a superconductor for an array of temperatures, given a critical
temperature, normal state resistance, elastic scattering time and a power law exponent for the 
temperature dependence of the phase breaking time.
"""
import time
import subprocess
import concurrent.futures
import numpy as np
import pandas as pd
from scipy.constants import pi,hbar,k,e,m_e
from typing import Tuple

def fscope_full_func(params: list|dict) -> list:
    """ Calculates the paraconductivity using the FSCOPE program

    Args:
        params (dict or list):
            Parameters to be passed to the FSCOPE program
            See the FSCOPE documentation for a list of possible parameters
    
    Returns:
        list: SC, sigma_AL, sigma_MTsum, sigma_MTint, sigma_DOS, sigma_DCR, sigma_tot
            SC is 0 or 1 for superconducting or normal state, sigma is the conductivity contribution
            in units of e^2/hbar, AL is the Aslamasov-Larkin contribution, MTsum and MTint are the
            contributions from the sum and integral parts of the Maki-Thompson contribution, DOS is
            the density of states contribution, DCR is the diffusion correction renormalisation
            contribution, and tot is the total fluctuation conductivity.
    """
    if isinstance(params, dict):
        params = [f'{k}={v}' for k,v in params.items()]
    output = subprocess.check_output(['FSCOPE/FSCOPE.exe']+params)
    lines = output.decode().splitlines()
    answer = [float(i) for i in lines[-1].split('\t')]
    return answer

def fscope_delta(T: float, Tc: float, tau: float, delta: float) -> list:
    """ Calculates fluctuation conductivity components for one temperature

    Args:
        T (float): Temperature in Kelvin
        Tc (float): Critical temperature in Kelvin
        tau (float): Elastic scattering time in seconds
        delta (float): Delta = pi * hbar / (8 * kB * T * tau_phi),
            parameterises the strength of phase breaking

    Returns:
        list: SC, sigma_AL, sigma_MTsum, sigma_MTint, sigma_DOS, sigma_DCR, sigma_tot
            in units of e^2/hbar, see fscope_full_func for a description of the components
    """
    t=T/Tc
    Tc0tau=Tc*tau*k/hbar
    params = ['ctype=100',
              f'tmin={t}',
              'dt=0.0',
              'Nt=1',
              'hmin=0.01',
              'dh=0.1',
              'Nh=1',
              f'Tc0tau={Tc0tau}',
              f'delta={delta}'
    ]
    return fscope_full_func(params)[2:]

def fscope_delta_wrapped(
    Ts: np.ndarray,
    Tc: float,
    tau: float,
    delta0: float,
    R0: float,
    alpha: float = -1
) -> Tuple[np.ndarray, pd.DataFrame]:
    """ Calculate the resistance vs temperature for given parameters

    tau_phi is given by tau_phi0 * T^alpha, with the default exponent being -1
    delta is then calculated as delta = pi * hbar / (8 * kB * T * tau_phi^alpha)
    delta0 is the value of delta at T = 1 K (i.e. delta0 = pi * hbar / (8 * kB * tau_phi0))

    Args:
        Ts (array): Temperatures in Kelvin
        Tc (float): Critical temperature in Kelvin
        tau (float): Elastic scattering time in seconds
        delta0 (float): Delta = pi * hbar / (8 * kB * tau_phi0)
            Parameterises the strength of the phase breaking
        R0 (float): Normal state resistance in Ohms
        alpha (float): Power law exponent for the temperature dependence of delta
    
    Returns:
        array: Resistance in Ohms for each temperature
        DataFrame: Components of the fluctuation conductivity and weak localisation for each T
    """
    results = np.zeros((8,len(Ts)))
    deltas=delta0*Ts**(-alpha-1)
    n=len(Ts)
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = np.zeros(n,dtype=object)
        for i in range(n):
            futures[i] = executor.submit(lambda x: fscope_delta(x,Tc,tau,deltas[i]), Ts[i])
            time.sleep(0.001)
    for i, future in enumerate(futures):
        results[0:7,i] = future.result()
        time.sleep(0.001)
    conversion = e**2/hbar
    sigma0=1/R0
    tauphi = pi*hbar/(8*k*Ts*deltas)
    results[1:7,:] *= conversion # convert to Siemens
    results[7,:] = weak_localisation(tau,tauphi)
    R = results[0,:]/(sigma0 + results[7,:] + results[6,:])
    # give column names to the results array
    results = pd.DataFrame(results.T,columns=['SC', 'AL', 'MTsum', 'MTint', 'DOS', 'DCR', 'Fluctuation_tot', 'WL'])
    results["MT"]=results["MTsum"]+results["MTint"]
    results["Total"]=results["Fluctuation_tot"]+results["WL"]
    return R, results

def weak_localisation(tau: float, tauphi: np.ndarray) -> np.ndarray:
    """ Calculates the weak localisation correction to the conductance

    Args:
        tau (float): Elastic scattering time in seconds
        tauphi (array): Phase breaking time in seconds
    
    Returns:
        array: The correction to conductance in Siemens (Ohms^-1)
    """
    dG = -e**2/(2*pi**2*hbar)*np.log(tauphi/tau)
    return dG

def calc_tau(
    rel_eff_mass: float,
    RN: float,
    Vg: float,
    Vg_n: np.ndarray,
    n: np.ndarray
) -> float:
    """ Calculates the elastic scattering time for a given backgate voltage

    Args:
        rel_eff_mass (float): Effective mass / electron rest mass
        RN (float): Normal state resistance in Ohms
        Vg (float): Backgate voltage in V
        Vg_n (array): Backgate voltages of measured carrier density in V, used to interpolate the 
            carrier density at Vg
        n (array): Carrier densities in m^-2, used to interpolate the carrier density at Vg

    Returns:
        float: The elastic scattering time in seconds
    """
    n_interpolated = np.interp(Vg,Vg_n,n)
    tau = rel_eff_mass*m_e/(n_interpolated*RN*e**2)
    return tau

def AL2D(
    Ts: np.ndarray,
    Tc: float,
    R0: float,
    C: float = e**2/(16*hbar)
) -> np.ndarray:
    """ Aslamasov-Larkin fluctuation conductance contribution

    Args:
        Ts (array): Temperatures in Kelvin
        Tc (float): Critical temperature in Kelvin
        R0 (float): Normal state resistance in Ohms
        C (float): Physical constant of e^2/16hbar, this is sometimes varied in literature

    Returns:
        array: Sheet conductance in Siemens (Ohms^-1).
    """
    return 1/(1/R0 + C/np.log(Ts/Tc)) * np.heaviside(Ts-Tc,0)
