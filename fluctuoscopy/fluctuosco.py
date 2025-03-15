"""Fluctuation conductivity of superconducting films in Python.

The function 'fscope' calculates the superconducting fluctuation and localization contributions
to the sheet conductance of a 2D superconducting film. The function uses SI units and returns
the total sheet resistance, useful for comparing with experimental data.

The calculations are based on the FSCOPE program, a C++ program written by Andreas Glatz. The
fscope uses a Rust port of the FSCOPE mc_sigma function including optimizations and
parallelization.

The function 'fscope_executable' provides a Python interface to the original FSCOPE program,
which has been compiled for Windows x86_64, Linux x86_64, MacOS x86_64 and arm64. If these do
not work on your system, you can compile the FSCOPE program from the source code available at
github.com/andreasglatz/FSCOPE, or github.com/g-kimbell/FSCOPE to include the shared library.

(c) 2025 Graham Kimbell, Ulderico Filipozzi, Andreas Glatz.
"""

from __future__ import annotations

import ctypes
import platform
import shlex
import subprocess
import warnings
from pathlib import Path

import numpy as np

from fluctuoscopy._fluctuoscopy import mc_sigma_parallel

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
    "ctype": "[integer] computation type",
    "tmin": "[float] : minimal temperature value in units of Tc0",
    "dt": "[float] : temperature interval in units of Tc",
    "Nt": "[integer] : number temperature value steps, i.e., t=tmin,tmin+dt,...,tmin+(Nt-1)*dt",
    "St": "[integer] : temperature scale: 0 - linear [default], 1 - log10, 2 - ln",
    "hmin": "[float] minimal dimensionless magnetic field h",
    "dh": "[float] magnetic field interval",
    "Nh": "[integer] number of magnetic field steps",
    "Sh": "[integer] magnetic field scale: 0 - linear [default], 1 - log10, 2 - ln",
    "vmin": "[float] minimal dimensionless voltage v",
    "dv": "[float] voltage interval",
    "Nv": "[integer] number of voltage steps",
    "Sv": "[integer] voltage scale: 0 - linear [default], 1 - log10, 2 - ln",
    "Tc0tau": "[float] value of Tc0*tau",
    "Tc0tauphi": "[float]  value of Tc0*tau_phi",
    "delta": "[float] value of delta=pi/(8*t*Tc0tauphi), if set overrides Tc0tauphi",
}

def get_fscope_executable() -> Path:
    """Get the path to the FSCOPE executable based on the operating system."""
    system = platform.system()
    base_dir = Path(__file__).resolve().parent
    if system == "Linux":
        return base_dir / "bin" / "FSCOPE_linux"
    if system == "Darwin":
        return base_dir / "bin" / "FSCOPE_mac"
    if system == "Windows":
        return base_dir / "bin" / "FSCOPE_windows.exe"
    msg = f"Unsupported operating system: {system}"
    raise RuntimeError(msg)

def get_fscope_lib() -> ctypes.CDLL | None:
    """Get the FSCOPE C library based on the operating system."""
    system = platform.system()
    base_dir = Path(__file__).resolve().parent
    if system == "Linux":
        shared_library_path = base_dir / "bin" / "fluctuoscope_extC.so"
    elif system == "Darwin":
        shared_library_path = base_dir / "bin" / "fluctuoscope_extC.dylib"
    elif system == "Windows":
        shared_library_path = base_dir / "bin" / "fluctuoscope_extC.dll"
    else:
        warnings.warn(
            f"Unsupported operating system: {system}, mc_sigma, hc2 and fscope_c functions will not work",
            stacklevel=2,
        )
        return None

    fscope_lib = ctypes.CDLL(str(shared_library_path))
    fscope_lib.MC_sigma_array.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
                                        ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
                                        ctypes.POINTER(ctypes.c_double), ctypes.c_int]
    fscope_lib.MC_sigma_array.restype = None
    fscope_lib.hc2_array.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.c_int]
    fscope_lib.hc2_array.restype = None
    return fscope_lib

# Get the path to the FSCOPE executable once at import time
FSCOPE_EXECUTABLE = get_fscope_executable()
FSCOPE_LIB = get_fscope_lib()

def mc_sigma(t: np.ndarray, h: np.ndarray, Tc_tau: np.ndarray, Tc_tauphi: np.ndarray) -> np.ndarray:
    """Calculate fluctuation conductivity components using the FSCOPE C library.

    Args:
        t (np.ndarray): Reduced temperature T/Tc
        h (np.ndarray): Reduced magnetic field H/Hc2 TODO is this actually correct
        Tc_tau (np.ndarray): Tc tau k_B / hbar (dimensionless)
        Tc_tauphi (np.ndarray): Tc tau_phi k_B / hbar (dimensionless)

    Returns:
        np.ndarray: sAL, sMTsum, sMTint, sDOS, sCC
            sAL: Aslamasov-Larkin contribution
            sMTsum: Maki-Thompson sum contribution
            sMTint: Maki-Thompson integral contribution
            sDOS: Density of states contribution
            sCC: Diffusion coefficient renormalisation contribution

    """
    results = np.zeros(5 * len(t), dtype=np.float64)
    if not FSCOPE_LIB:
        msg = "FSCOPE C library not available on this operating system"
        raise NotImplementedError(msg)
    FSCOPE_LIB.MC_sigma_array(
        t.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        h.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        Tc_tau.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        Tc_tauphi.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        results.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        len(t),
    )
    return results.reshape((len(t),5)).T

def mc_sigma_rust(t: np.ndarray, h: np.ndarray, Tc_tau: np.ndarray, Tc_tauphi: np.ndarray) -> dict:
    """Calculate fluctuation conductivity components using the FSCOPE Rust library.

    Args:
        t (np.ndarray): Reduced temperature T/Tc
        h (np.ndarray): Reduced magnetic field H/Hc2 TODO is this actually correct
        Tc_tau (np.ndarray): Tc tau k_B / hbar (dimensionless)
        Tc_tauphi (np.ndarray): Tc tau_phi k_B / hbar (dimensionless)

    Returns:
        dict: al, mtsum, mtint, dos, dcr in units of G0 (NOT SI)
            al: Aslamasov-Larkin contribution
            mtsum: Maki-Thompson sum contribution
            mtint: Maki-Thompson integral contribution
            dos: Density of states contribution
            dcr: Diffusion coefficient renormalisation contribution

    """
    return mc_sigma_parallel(t, h, Tc_tau, Tc_tauphi)

def hc2(t: np.ndarray) -> np.ndarray:
    """Calculate the upper critical field using the FSCOPE C library."""
    if not isinstance(t, np.ndarray):
        t = np.array(t)
    results = np.zeros(len(t), dtype=np.float64)
    if not FSCOPE_LIB:
        msg = "FSCOPE C library not available on this operating system"
        raise NotImplementedError(msg)
    FSCOPE_LIB.hc2_array(
        t.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        results.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        len(t),
    )
    return results

def fscope_executable(params: dict) -> list:
    """Calculate the paraconductivity using the FSCOPE program.

    Args:
        params (dict or list):
            Parameters to be passed to the FSCOPE program
            See the FSCOPE documentation for a list of possible parameters

    Returns:
        list: lines of output from the FSCOPE program

    """
    if "ctype" not in params:
        message = "\n".join(
            [
                "No computation type specified",
                "Include 'ctype' in the params dictionary with one of the following values:",
            ] + [f"{key}: {value}" for key, value in KNOWN_CTYPES.items()],
        )
        raise ValueError(message)
    # if ctype not known
    unknown_params = [k for k in params if k not in KNOWN_PARAMS]
    if unknown_params:
        message = "\n".join([
            "Unknown parameters:", "Please use the following keys for the params dictionary:", *unknown_params,
        ])
        raise ValueError(message)
    # Sanitize and prepare the command arguments
    command = [FSCOPE_EXECUTABLE] + [f"{k}={shlex.quote(str(v))}" for k, v in params.items()]
    output = subprocess.check_output(command)
    return output.decode().splitlines()

def fscope_full(params: dict | None = None) -> dict:
    """Calculate paraconductivity using the FSCOPE executable program.

    Args:
        params (dict):
            Parameters to be passed to the FSCOPE program
            See the FSCOPE documentation for a list of possible parameters
            Or run without parameters to see the usage message

    Returns:
        dict: Dictionary of the output from the FSCOPE program, with header
            and separate data columns

    """
    if params is None:
        params = {}
    output = fscope_executable(params)
    if "FLUCTUOSCOPE" in output[0]:
        message = "\n".join(
            [
                "Usage of fluctuoscopy, using FLUCTUOSCOPE version 2.1:",
                "Add params to the function as a dictionary using the following keys:",
            ] + output[5:],
        )
        raise ValueError(message)

    result = {}
    header = []
    data: list | np.ndarray = []
    for line in output:
        if line.startswith("#"):
            header += [line.strip("#")]
        else:
            data += [[float(n) for n in line.split("\t")]]
    col_names = header[-1].split("\t")
    data = np.array(data).T
    result["header"] = header[:-1]
    result.update(dict(zip(col_names, data)))
    return result

def fscope(
    Ts: float | np.ndarray,
    Tc: float | np.ndarray,
    tau: float | np.ndarray,
    tau_phi0: float | np.ndarray,
    R0: float | np.ndarray,
    alpha: float | np.ndarray = -1.0,
    tau_SO: float | np.ndarray | None = None,
) -> tuple[np.ndarray, dict]:
    """Get resistance, fluctuation and localization contributions."""
    t = Ts/Tc
    Tc_tau = Tc*tau*k/hbar
    tau_phi = tau_phi0*Ts**alpha
    Tc_tauphi = Tc*tau_phi*k/hbar

    # make sure all inputs are arrays, find the longest array
    lens = set()
    for val in locals().values():
        if isinstance(val, np.ndarray):
            lens |= {len(val)}
    if len(lens) > 1:
        msg = "All input arrays must have the same length"
        raise ValueError(msg)
    max_len = 1 if len(lens) == 0 else lens.pop()
    if not isinstance(t, np.ndarray):
        t = np.full(max_len, t)
    if not isinstance(Tc_tau, np.ndarray):
        Tc_tau = np.full(max_len, Tc_tau)
    if not isinstance(tau_phi, np.ndarray):
        tau_phi = np.full(max_len, tau_phi)
    if not isinstance(Tc_tauphi, np.ndarray):
        Tc_tauphi = np.full(max_len, Tc_tauphi)
    h = np.full(max_len, 0.01)

    # Fluctuation components in units of G0
    results = mc_sigma_rust(t, h, Tc_tau, Tc_tauphi)
    results = {key: np.array(val) for key, val in results.items()}
    fluc_total = (results["al"] + results["mtsum"] + results["mtint"] + results["dos"] + results["dcr"])
    # WL and WAL already in Ohms^-1
    WL = weak_localization(tau,tau_phi)
    WAL = weak_antilocalization(tau_SO,tau_phi) if tau_SO is not None else np.zeros(max_len)
    conversion = e**2/hbar
    sigma0=1/R0
    R: np.ndarray = ~results["sc"]/(sigma0 + fluc_total*conversion + WL + WAL)
    results_dict = {
        "AL": results["al"]*conversion,
        "MTsum": results["mtsum"]*conversion,
        "MTint": results["mtint"]*conversion,
        "MT": (results["mtsum"] + results["mtint"])*conversion,
        "DOS": results["dos"]*conversion,
        "DCR": results["dcr"]*conversion,
        "Fluctuation_tot": fluc_total*conversion,
        "WL": WL,
        "WAL": WAL,
        "SC": results["sc"],
        "Total": fluc_total*conversion + WL + WAL,
    }
    return R, results_dict

def fscope_c(
    Ts: np.ndarray,
    Tc: float,
    tau: float,
    tau_phi0: float,
    R0: float,
    alpha: float = -1.0,
    tau_SO: float | np.ndarray | None = None,
) -> tuple[np.ndarray, dict]:
    """Get resistance, fluctuation and localization contributions from T.

    Units are SI (Ohm, seconds). Uses the parallelized C library for faster computation.

    Args:
        Ts (np.ndarray): Temperatures in Kelvin
        Tc (float): Critical temperature in Kelvin
        tau (float): Elastic scattering time in seconds
        tau_phi0 (float): Phase breaking time in seconds
        R0 (float): Normal state resistance in Ohms
        alpha (float): Exponent for phase breaking time temperature dependence
        tau_SO (float, np.ndarray, optional): Spin-orbit scattering time in seconds

    Returns:
        tuple[np.ndarray, dict]: Resistance in Ohms and dictionary of contributions

    """
    if not all(Ts > Tc):
        msg = "All temperatures must be above the critical temperature"
        raise ValueError(msg)
    results = np.zeros((9,len(Ts)))
    t = np.array(Ts)/Tc
    h = np.zeros(len(Ts))+0.01
    Tc_tau = np.array([Tc*tau*k/hbar]*len(Ts))
    tau_phi = tau_phi0*np.array(Ts)**alpha
    Tc_tauphi = Tc*tau_phi*k/hbar
    # Fluctuation components in units of G0
    results = mc_sigma(t, h, Tc_tau, Tc_tauphi)
    fluc_total = np.sum(results,axis=0)
    # WL and WAL already in Ohms^-1
    WL = weak_localization(tau,tau_phi)
    WAL = weak_antilocalization(tau_SO,tau_phi) if tau_SO is not None else np.zeros(len(Ts))
    conversion = e**2/hbar
    sigma0=1/R0
    R = 1/(sigma0 + fluc_total*conversion + WL + WAL)
    results_dict = {
        "AL": results[0]*conversion,
        "MTsum": results[1]*conversion,
        "MTint": results[2]*conversion,
        "DOS": results[3]*conversion,
        "DCR": results[4]*conversion,
        "Fluctuation_tot": fluc_total*conversion,
        "WL": WL,
        "WAL": WAL,
        "MT": (results[1] + results[2])*conversion,
        "Total": fluc_total*conversion + WL + WAL,
    }
    return R, results_dict

def weak_localization(tau: float | np.ndarray, tau_phi: float | np.ndarray) -> np.ndarray:
    """Calculate the weak localization correction to the conductance.

    Args:
        tau (float): Elastic scattering time in seconds
        tau_phi (array): Phase breaking time in seconds

    Returns:
        array: The correction to conductance in Siemens (Ohms^-1)

    """
    return np.array(-e**2/(2*pi**2*hbar)*np.log(tau_phi/tau))

def weak_antilocalization(tau_SO: float | np.ndarray, tau_phi: np.ndarray) -> np.ndarray:
    """Calculate the weak antilocalization correction to the conductance.

    Args:
        tau_SO (float, np.ndarray): Spin-orbit scattering time in seconds
        tau_phi (array): Phase breaking time in seconds

    Returns:
        array: The correction to conductance in Siemens (Ohms^-1)

    """
    return e**2 / (2*pi**2*hbar) * np.log((1+tau_phi/tau_SO) * (1+2*tau_phi/tau_SO)**0.5)

def AL2D(
    Ts: np.ndarray,
    Tc: float,
    R0: float,
    C: float = e**2/(16*hbar),
) -> np.ndarray:
    """Aslamasov-Larkin fluctuation conductance contribution.

    Args:
        Ts (array): Temperatures in Kelvin
        Tc (float): Critical temperature in Kelvin
        R0 (float): Normal state resistance in Ohms
        C (float): Physical constant of e^2/16hbar, this is sometimes varied in literature

    Returns:
        array: Sheet conductance in Siemens (Ohms^-1)

    """
    return 1/(1/R0 + C/np.log(Ts/Tc)) * np.heaviside(Ts-Tc,0)
