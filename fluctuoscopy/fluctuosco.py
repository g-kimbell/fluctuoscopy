"""
This module provides a python interface to the FSCOPE program, which calculates fluctuation
conductivity components in superconductors under various conditions.

The most important function is fscope, which takes a dictionary of parameters and returns a
dictionary of the output from the FSCOPE program.

We also include a parallelised function for calculating fluctuation conductivity components
in the absence of magnetic field, as well as functions for weak (anti)localization corrections.
"""

import subprocess
import os
import platform
from typing import Tuple
import numpy as np
from multiprocess import Pool, cpu_count
import ctypes
import warnings

pi = np.pi
hbar=1.0545718176461565e-34
k=1.380649e-23
e=1.602176634e-19
m_e=9.1093837015e-31
KNOWN_CTYPES = {
    1: "output hc2(t) for t=0..1 using Nt t-steps (plus approximation)",
    100: "full fluctuation conductivity calculation using t,h",
    111: "full fluctuation conductivity parallel to hc2(t), use parameter hc2s>1 and Nt",
    200: "tunnel IV using t,h,v (should fix one Nx to 1 or get 4D data)",
    201: "tunnel conductance using t,h,v (should fix one Nx to 1 or get 4D data)",
    202: "zero bias tunnel conductance using t,h",
    211: "tunnel conductance parallel to hc2(t), use parameter hc2s>1 and Nt, v",
    290: "test",
    300: "Nernst beta_xy using t,h",
    390: "test",
    400: "NMR 1/T1 normalized to Karringaand Gi(2) using t,h",
    410: "as 400, but along hc2(t) uses Nt only and hc2s>1",
    403: "3D NMR 1/T1 normalized to Karringaand Gi(2) using t,h and aniso (r)",
    413: "as 403, but along hc2(t) uses Nt only and hc2s>1",
    500: "susceptibility t,h",
}
KNOWN_PARAMS = {
    'ctype': "[integer] computation type",
    'tmin': "[float] : minimal temperature value in units of Tc0",
    'dt': "[float] : temperature interval in units of Tc",
    'Nt': "[integer] : number temperature value steps, i.e., t=tmin,tmin+dt,...,tmin+(Nt-1)*dt",
    'St': "[integer] : temperature scale: 0 - linear [default], 1 - log10, 2 - ln",
    'hmin': "[float] minimal dimensionless magnetic field h",
    'dh': "[float] magnetic field interval",
    'Nh': "[integer] number of magnetic field steps",
    'Sh': "[integer] magnetic field scale: 0 - linear [default], 1 - log10, 2 - ln",
    'vmin': "[float] minimal dimensionless voltage v",
    'dv': "[float] voltage interval",
    'Nv': "[integer] number of voltage steps",
    'Sv': "[integer] voltage scale: 0 - linear [default], 1 - log10, 2 - ln",
    'Tc0tau': "[float] value of Tc0*tau",
    'Tc0tauphi': "[float]  value of Tc0*tau_phi",
    'delta': "[float] value of delta=pi/(8*t*Tc0tauphi), if set overrides Tc0tauphi",
}

def get_fscope_executable() -> str:
    system = platform.system()
    base_dir = os.path.dirname(os.path.abspath(__file__))
    if system == 'Linux':
        return os.path.join(os.path.dirname(__file__), 'bin', 'FSCOPE_linux')
    if system == 'Darwin':
        return os.path.join(os.path.dirname(__file__), 'bin', 'FSCOPE_mac')
    if system == 'Windows':
        return os.path.join(base_dir, 'bin', 'FSCOPE_windows.exe')
    raise RuntimeError(f"Unsupported operating system: {system}")

def get_fscope_lib() -> ctypes.CDLL:
    system = platform.system()
    base_dir = os.path.dirname(os.path.abspath(__file__))
    if system == 'Linux':
        return os.path.join(os.path.dirname(__file__), 'bin', 'fluctuoscope_extC.so')
    elif system == 'Darwin':
        # return os.path.join(os.path.dirname(__file__), 'bin', 'libFSCOPE.dylib')
        warnings.warn("FSCOPE C library not available on MacOS, mc_sigma, hc2 and fscope_R functions will not work")
        return None
    elif system == 'Windows':
        dll_path = os.path.join(base_dir, 'bin', 'fluctuoscope_extC.dll')
        fscope_lib = ctypes.CDLL(dll_path)
        fscope_lib.MC_sigma_array.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
                                            ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
                                            ctypes.POINTER(ctypes.c_double), ctypes.c_int]
        fscope_lib.MC_sigma_array.restype = None
        fscope_lib.hc2_array.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.c_int]
        fscope_lib.hc2_array.restype = None
        return fscope_lib
    else:
        warnings.warn(f"Unsupported operating system: {system}, mc_sigma, hc2 and fscope_R functions will not work")
        return None

# Get the path to the FSCOPE executable once at import time
FSCOPE_EXECUTABLE = get_fscope_executable()
FSCOPE_LIB = get_fscope_lib()

def mc_sigma(t: np.ndarray, h: np.ndarray, Tc_tau: np.ndarray, Tc_tauphi: np.ndarray) -> np.ndarray:
    """Calculate fluctuation conductivity components using the FSCOPE C library.

    Args:
        t (np.ndarray): Reduced temperature T/Tc
        h (np.ndarray): Reduced magnetic field H/Hc2 TODO is this actually correct
        Tc_tau (np.ndarray): Tc*tau*k/hbar (dimensionless)
        Tc_tauphi (np.ndarray): Tc*tau_phi*k/hbar (dimensionless)

    Returns:
        np.ndarray: sAL, sMTsum, sMTint, sDOS, sCC
            sAL: Aslamasov-Larkin contribution
            sMTsum: Maki-Thompson sum contribution
            sMTint: Maki-Thompson integral contribution
            sDOS: Density of states contribution
            sCC: Diffusion coefficient renormalisation contribution

    """
    results = np.zeros(5 * len(t), dtype=np.float64)
    FSCOPE_LIB.MC_sigma_array(
        t.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        h.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        Tc_tau.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        Tc_tauphi.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        results.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        len(t),
    )
    return results.reshape((len(t),5)).T

def hc2(t: np.ndarray) -> np.ndarray:
    if not isinstance(t, np.ndarray):
        t = np.array(t)
    results = np.zeros(len(t), dtype=np.float64)
    FSCOPE_LIB.hc2_array(
        t.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        results.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        len(t),
    )
    return results

def fscope_full_func(params: dict) -> list:
    """ Calculates the paraconductivity using the FSCOPE program

    Args:
        params (dict or list):
            Parameters to be passed to the FSCOPE program
            See the FSCOPE documentation for a list of possible parameters
    
    Returns:
        list: lines of output from the FSCOPE program
    """
    if 'ctype' not in params.keys():
        message = [
            "No computation type specified",
            "Include 'ctype' in the params dictionary with one of the following values:",
        ] + [f"{key}: {value}" for key, value in KNOWN_CTYPES.items()]
        message = "\n".join(message)
        raise ValueError(message)
    # if ctype not known
    unknown_params = [k for k in params if k not in KNOWN_PARAMS]
    if unknown_params:
        message = [
            "Unknown parameters:",
            "Please use the following keys for the params dictionary:",
        ] + unknown_params
        message = "\n".join(message)
        raise ValueError(message)
    output = subprocess.check_output([FSCOPE_EXECUTABLE]+[f'{k}={v}' for k,v in params.items()])
    output_decoded = output.decode().splitlines()
    return output_decoded

def fscope(params: dict = None) -> dict:
    """ Calculates the paraconductivity using the FSCOPE program

    Args:
        params (dict or list):
            Parameters to be passed to the FSCOPE program
            See the FSCOPE documentation for a list of possible parameters
            Or run without parameters to see the usage message

    Returns:
        dict: Dictionary of the output from the FSCOPE program, with header
            and separate data columns
    """
    if params is None:
        params = {}
    output = fscope_full_func(params)
    if 'FLUCTUOSCOPE' in output[0]:
        message = [
            "Usage of fluctuoscopy, using FLUCTUOSCOPE version 2.1:",
            "Add params to the function as a dictionary using the following keys:",
        ] + output[5:]
        message = "\n".join(message)
        raise ValueError(message)

    result = {}
    header = []
    data = []
    for line in output:
        if line.startswith('#'):
            header += [line.strip('#')]
        else:
            data += [[float(n) for n in line.split('\t')]]
    col_names = header[-1].split('\t')
    data = np.array(data).T
    result['header'] = header[:-1]
    for col_name, col_data in zip(col_names, data):
        result[col_name] = col_data
    return result

def fscope_parallel(params: dict) -> dict:
    """ Calculates the paraconductivity using the FSCOPE program in parallel

    Args:
        params (dict):
            Parameters to be passed to the FSCOPE program
            Can give an array of values for each parameter to calculate in parallel
    
    Returns:
        dict: Dictionary of the output
    """
    # Convert all parameters to lists
    for key in params:
        if not isinstance(params[key], list):
            params[key] = [params[key]]
    # Check that all lists are the same length or length 1
    lengths = [len(params[key]) for key in params]
    print(lengths)
    print(set(lengths))
    if len(set(lengths)) > 2:
        raise ValueError("All parameters must be a list of the same length, or a single value")
    # Make all lists the same length
    n = max(lengths)
    for key in params:
        if len(params[key]) == 1:
            params[key] = params[key]*n
    # Run the parallel calculation
    with Pool(max(1,cpu_count()-1)) as pool:
        arg_list = [{key: params[key][i] for key in params} for i in range(n)]
        results = pool.map(fscope, arg_list)
    # Combine the results
    result = {}
    for key in results[0]:
        if key != 'header':
            result[key] = np.concatenate([r[key] for r in results])
    return result

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
            SC is 0 or 1 for superconducting or normal state, sigma is the conductivity contribution
            in units of e^2/hbar, AL is the Aslamasov-Larkin contribution, MTsum and MTint are the
            contributions from the sum and integral parts of the Maki-Thompson contribution, DOS is
            the density of states contribution, DCR is the diffusion correction renormalisation
            contribution, and tot is the total fluctuation conductivity.
    """
    t=T/Tc
    Tc0tau=Tc*tau*k/hbar
    params = {
        'ctype': 100,
        'tmin': t,
        'dt': 0.0,
        'Nt': 1,
        'hmin': 0.01,
        'dh': 0.1,
        'Nh': 1,
        'Tc0tau': Tc0tau,
        'delta': delta
    }
    output = fscope_full_func(params)
    result = [float(i) for i in output[-1].split('\t')]
    return result[2:]

def fscope_R(
    Ts: np.ndarray,
    Tc: float,
    tau: float,
    tau_phi0: float,
    R0: float,
    alpha: float = -1,
    tau_SO: float = None,
) -> Tuple[np.ndarray, dict]:
    """Get resistance, fluctuation and localization contributions from T.

    Units are SI (Ohm, seconds). Uses the parallelized C library for faster computation.
    """
    assert all(Ts > Tc), "All temperatures must be above the critical temperature"
    results = np.zeros((9,len(Ts)))
    t = np.array(Ts)/Tc
    h = np.zeros(len(Ts))+0.01
    Tc_tau = np.array([Tc*tau*k/hbar]*len(Ts))
    tau_phi = tau_phi0*np.array(Ts)**alpha
    Tc_tauphi = Tc*tau_phi*k/hbar
    results = mc_sigma(t, h, Tc_tau, Tc_tauphi)
    fluc_total = np.sum(results,axis=0)
    WL = weak_localization(tau,tau_phi) # already in Ohms^-1
    WAL = weak_antilocalization(tau_SO,tau_phi) if tau_SO else np.zeros(len(Ts)) # already in Ohms^-1
    conversion = e**2/hbar
    sigma0=1/R0
    R = 1/(sigma0 + fluc_total*conversion + WL + WAL)
    results_dict = {
        'AL': results[0]*conversion,
        'MTsum': results[1]*conversion,
        'MTint': results[2]*conversion,
        'DOS': results[3]*conversion,
        'DCR': results[4]*conversion,
        'Fluctuation_tot': fluc_total*conversion,
        'WL': WL,
        'WAL': WAL,
        'MT': (results[1] + results[2])*conversion,
        'Total': fluc_total*conversion + WL + WAL,
    }
    return R, results_dict

def fscope_delta_wrapped(
    Ts: np.ndarray,
    Tc: float,
    tau: float,
    delta0: float,
    R0: float,
    alpha: float = -1,
    tau_SO: float = None
) -> Tuple[np.ndarray, dict]:
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
        tau_SO (float, optional): Spin-orbit scattering time in seconds, used for weak 
            antilocalization correction
    
    Returns:
        array: Resistance in Ohms for each temperature
        dict: Components of the fluctuation conductivity and weak localization for each T
    """
    results = np.zeros((9,len(Ts)))
    deltas=delta0*Ts**(-alpha-1)
    n=len(Ts)

    def fscope_delta_wrapper(args):
        x, Tc, tau, delta = args
        return fscope_delta(x, Tc, tau, delta)
    with Pool(max(1,cpu_count()-1)) as pool:
        args_list = [(Ts[i], Tc, tau, deltas[i]) for i in range(n)]
        results_list = pool.map(fscope_delta_wrapper, args_list)
    results[0:7, :] = np.array(results_list).T
    conversion = e**2/hbar
    sigma0=1/R0
    tau_phi = pi*hbar/(8*k*Ts*deltas)
    results[1:7,:] *= conversion # convert to Siemens
    results[7,:] = weak_localization(tau,tau_phi)
    if tau_SO:
        results[8,:] = weak_antilocalization(tau_SO,tau_phi)
    R = results[0,:]/(sigma0 + results[6,:] + results[7,:] + results[8,:])
    # give column names to the results array
    results_dict = {
        'SC': results[0],
        'AL': results[1],
        'MTsum': results[2],
        'MTint': results[3],
        'DOS': results[4],
        'DCR': results[5],
        'Fluctuation_tot': results[6],
        'WL': results[7],
        'WAL': results[8]
    }
    results_dict["MT"] = results_dict["MTsum"] + results_dict["MTint"]
    results_dict["Total"] = results_dict["Fluctuation_tot"] + results_dict["WL"] + results_dict["WAL"]
    return R, results_dict

def weak_localization(tau: float, tau_phi: np.ndarray) -> np.ndarray:
    """ Calculates the weak localization correction to the conductance

    Args:
        tau (float): Elastic scattering time in seconds
        tau_phi (array): Phase breaking time in seconds
    
    Returns:
        array: The correction to conductance in Siemens (Ohms^-1)
    """
    dG = -e**2/(2*pi**2*hbar)*np.log(tau_phi/tau)
    return dG

def weak_antilocalization(tau_SO: float, tau_phi: np.ndarray) -> np.ndarray:
    """ Calculates the weak antilocalization correction to the conductance

    Args:
        tau_SO (float): Spin-orbit scattering time in seconds
        tau_phi (array): Phase breaking time in seconds
    
    Returns:
        array: The correction to conductance in Siemens (Ohms^-1)
    """
    dG = e**2/(2*pi**2*hbar)*np.log(
        (1+tau_phi/tau_SO) *
        (1+2*tau_phi/tau_SO)**0.5
    )
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
