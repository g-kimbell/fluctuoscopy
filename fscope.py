"""This module provides functions to calculate the fluctuation conductivity of a 2-dimensional
superconductor in the absence of magnetic field using the FSCOPE C++ program. The functions are 
multi-threaded and wrapped in a way that allows easier use of e.g. least squares fitting algorithms.

We also include the effects of weak localisation and a varying elastic scattering time with
backgate to simplify the common process of calculating sheet resistance in a field-effect
device.

The module requires the FSCOPE program to be compiled and placed in the same directory as this file.

The most important function is 'fscope_delta_wrapped', which calculates the resistance of a
superconductor as a function of temperature, critical temperature, normal state resistance, elastic
scattering time and a power law exponent for the temperature dependence of the phase breaking time.
"""
import time
import subprocess
import concurrent.futures
import numpy as np
from scipy.constants import pi,hbar,k,e,m_e

def fscope_full_func(params):
    """Calculates the paraconductivity of a superconductor using the FSCOPE program.
    
    Args:
        params (dict or list): parameters to be passed to the FSCOPE program.
            See the FSCOPE documentation for a list of possible parameters.
        
    Returns:
        list: A list of the paraconductivity values for the given parameters.
    """
    if isinstance(params, dict):
        params = [f'{k}={v}' for k,v in params.items()]
    output = subprocess.check_output(['FSCOPE/FSCOPE.exe']+params)
    answer = str(output).split('\\n')[-2][:-2]
    answer = answer.split('\\t')
    answer = [float(i) for i in answer]
    return answer

def fscope_delta(T,Tc,tau,delta):
    """Calculates the different components of fluctuation conductivity for one temperature with
    given parameters.

    Args:
        T (float): temperature in Kelvin.
        Tc (float): critical temperature in Kelvin.
        tau (float): elastic scattering time in seconds.
        delta (float): delta = pi * hbar / (8 * kB * T * tau_phi)
                       paramaterises the strength of the phase breaking

    Returns:
        float: The paraconductivity value for the given parameters.
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

def fscope_delta_wrapped(Ts,Tc,tau,delta0,R0,alpha=-1):
    """Calculate the resistance vs temperature for given parameters.
    tau_phi is given by tau_phi0 * T^alpha, with the default exponent being -1.
    delta is then calculated as delta = pi * hbar / (8 * kB * T * tau_phi^alpha)
    delta0 is the value of delta at T = 1 K (i.e. delta0 = pi * hbar / (8 * kB * tau_phi0))
    
    Args:
        Ts (array): temperatures in K.
        Tc (float): critical temperature in K.
        tau (float): elastic scattering time in s.
        delta0 (float): pi * hbar / (8 * kB * 1 * tau_phi)
                        i.e the delta value at T = 1 K
        R0 (float): normal state resistance in Ohms.
        alpha (float): power law exponent for the temperature dependence of delta.
    
    Returns:
        array: The resistance values in Ohms for the given temperatures.
    """
    results = np.zeros((len(Ts),7))
    deltas=delta0*Ts**(-alpha-1)
    n=len(Ts)
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = np.zeros(n,dtype=object)
        for i in range(n):
            futures[i] = executor.submit(lambda x: fscope_delta(x,Tc,tau,deltas[i]), Ts[i])
            time.sleep(0.001)
    for i, future in enumerate(futures):
        results[i,:] = future.result()
        time.sleep(0.001)
    conversion = e**2/hbar
    sigma0=1/R0
    tauphi = pi*hbar/(8*k*Ts*deltas)
    R = results[:,0]/(sigma0 + weak_localisation(tau,tauphi) + results[:,6]*conversion)
    return R

def weak_localisation(tau,tauphi):
    """Calculates the weak localisation correction to the conductance.
    
    Args:
        tau (float): elastic scattering time in s.
        tauphi (array): phase breaking time in s.
    
    Returns:
        array: The correction to conductance in Siemens (Ohms^-1).
    """
    dG = -e**2/(2*pi**2*hbar)*np.log(tauphi/tau)
    return dG

def calc_tau(relative_effective_mass,RN,Vg,Vg_n,n):
    """Calculates the elastic scattering time for a given backgate voltage.

    Args:
        relative_effective_mass (float): effective mass / electron rest mass.
        RN (float): normal state resistance in Ohms.
        Vg (float): backgate voltage in V.
        Vg_n (array): backgate voltages of measured carrier density in V.
        n (array): carrier densities in m^-2.

    Returns:
        float: The elastic scattering time in seconds.
    """
    n_interpolated = np.interp(Vg,Vg_n,n)
    tau = relative_effective_mass*m_e/(n_interpolated*RN*e**2)
    return tau

def AL2D(Ts,Tc,R0,C=e**2/(16*hbar)):
    """Aslamasov-Larkin fluctuation conductance contribution
    
    Args:
        Ts (array): temperatures in Kelvin.
        Tc (float): critical temperature in Kelvin.
        R0 (float): normal state resistance in Ohms.
        C (float): A physical constant of e^2/16hbar, this is sometimes varied in literature.
    
    Returns:
        array: Sheet resistance in Ohms.
    """
    return 1/(1/R0 + C/np.log(Ts/Tc))
