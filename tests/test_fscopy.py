"""Tests for fluctuoscopy.fluctuosco.py module."""
import unittest

import numpy as np
import pytest

from fluctuoscopy.fluctuosco import (
    AL2D,
    fscope,
    fscope_c,
    fscope_executable,
    fscope_full,
    mc_sigma,
    mc_sigma_rust,
    weak_antilocalization,
    weak_localization,
)

e = 1.60217662e-19
m_e = 9.10938356e-31
hbar = 1.0545718e-34
k_B = 1.38064852e-23
pi = np.pi

class TestFscopeFullFunc(unittest.TestCase):
    """Tests for fscope_executable function."""

    def test_fscope_executable_empty(self) -> None:
        """Test fscope_executable with empty parameters."""
        with pytest.raises(ValueError):
            fscope_executable({})

    def test_fscope_executable_basic(self) -> None:
        """Test fscope_executable with one set of input parameters."""
        params = {
            "ctype": 100,
            "tmin": 2,
            "dt": 0,
            "Nt": 1,
            "hmin": 0.01,
            "dh": 0,
            "Nh": 1,
            "Tc0tau": 1e-2,
            "delta": 1e-3,
        }
        output = fscope_executable(params)
        result = [float(r) for r in output[-1].split("\t")]
        expected = [
            2.0,
            0.01,
            1.0,
            0.013652032863547285,
            -0.04670521178360401,
            0.4186962142931441,
            -0.015732975271415026,
            0.0030605431719476525,
            0.37297060327361997,
        ]
        np.testing.assert_array_almost_equal(result, expected, decimal=6)

class TestFscope(unittest.TestCase):
    """Tests for fscope function."""

    def test_fscope_empty(self) -> None:
        """Test fscope with empty parameters."""
        with pytest.raises(ValueError):
            fscope_full({})

    def test_fscope_basic(self) -> None:
        """Test fscope with one set of input parameters."""
        params = {
            "ctype": 100,
            "tmin": 2,
            "dt": 0,
            "Nt": 1,
            "hmin": 0.01,
            "dh": 0,
            "Nh": 1,
            "Tc0tau": 1e-2,
            "delta": 1e-3,
        }
        result = fscope_full(params)
        expected = { # ignore header
            "t": np.array([2.]),
            "h": np.array([0.01]),
            "SC": np.array([1.]),
            "sAL": np.array([0.01365203]),
            "sMTsum": np.array([-0.04670521]),
            "sMTint": np.array([0.41869621]),
            "sDOS": np.array([-0.01573298]),
            "sDCR": np.array([0.00306054]),
            "sigma": np.array([0.3729706]),
        }
        for key, value in expected.items():
            np.testing.assert_array_almost_equal(result[key], value, decimal=6)

class TestWeakLocalization(unittest.TestCase):
    """Tests for weak_localization function."""

    def test_weak_localization_basic(self) -> None:
        """Test weak_localization with one set of parameters."""
        tau = 1e-12
        tau_phi = np.array([1e-11, 2e-11, 3e-11])
        expected = -e**2 / (2 * np.pi**2 * hbar) * np.log(tau_phi / tau)
        result = weak_localization(tau, tau_phi)
        np.testing.assert_array_almost_equal(result, expected, decimal=6)

class TestWeakAntilocalization(unittest.TestCase):
    """Tests for weak_antilocalization function."""

    def test_weak_antilocalization_basic(self) -> None:
        """Test weak_antilocalization with one set of parameters."""
        tau_SO = 1e-12
        tau_phi = np.array([1e-11, 2e-11, 3e-11])
        expected = e**2 / (2 * np.pi**2 * hbar) * np.log((1 + tau_phi / tau_SO) * (1 + 2 * tau_phi / tau_SO)**0.5)
        result = weak_antilocalization(tau_SO, tau_phi)
        np.testing.assert_array_almost_equal(result, expected, decimal=6)

class TestAL2D(unittest.TestCase):
    """Tests for AL2D function."""

    def test_al2d_basic(self) -> None:
        """Test AL2D with basic parameters."""
        Ts = np.array([1.5, 2.0, 2.5])
        Tc = 1.0
        R0 = 10.0
        expected = 1 / (1 / R0 +  e**2 / (16*hbar) / np.log(Ts / Tc)) * np.heaviside(Ts - Tc, 0)
        result = AL2D(Ts, Tc, R0)
        np.testing.assert_array_almost_equal(result, expected, decimal=6)

    def test_al2d_with_custom_C(self) -> None:
        """Test AL2D with custom C."""
        Ts = np.array([1.5, 2.0, 2.5])
        Tc = 1.0
        R0 = 10.0
        C = 1.0
        expected = 1 / (1 / R0 + C / np.log(Ts / Tc)) * np.heaviside(Ts - Tc, 0)
        result = AL2D(Ts, Tc, R0, C)
        np.testing.assert_array_almost_equal(result, expected, decimal=6)

    def test_al2d_below_tc(self) -> None:
        """Test AL2D with Ts below Tc."""
        Ts = np.array([1.0, 1.5, 1.9])
        Tc = 2.0
        R0 = 10.0
        expected = np.zeros_like(Ts)
        result = AL2D(Ts, Tc, R0)
        np.testing.assert_array_almost_equal(result, expected, decimal=6)

class TestFscopeFluc(unittest.TestCase):
    """Tests for fscope_c function."""

    def test_fscope_c(self) -> None:
        """Test fscope_c with basic parameters."""
        Ts = np.array([1.5, 2.0, 2.5])
        Tc = 1.0
        tau = 1e-12
        delta0 = 1e-3
        tauphi0 = pi*hbar/(8*k_B*delta0)
        R0 = 1000.0
        alpha = -1
        tau_SO = None

        expected_R = [946.40032506, 1023.33984024, 1049.08408124]
        expected_results = {
            "AL": np.array([1.31040170e-06, 1.43936402e-07, 2.99469748e-08]),
            "MTsum": np.array([-4.04100310e-06, -1.65381459e-06, -8.44054498e-07]),
            "MTint": np.array([1.55243615e-04, 6.95601598e-05, 4.17476473e-05]),
            "DOS": np.array([-2.20475627e-06, -6.91236851e-07, -2.96615333e-07]),
            "DCR": np.array([5.54148708e-08, 1.42590054e-08, 4.65064346e-09]),
            "Fluctuation_tot": np.array([1.50363672e-04, 6.73733038e-05, 4.06415750e-05]),
            "WL": np.array([-9.37283634e-05, -9.01808203e-05, -8.74291320e-05]),
            "WAL": np.array([0., 0., 0.]),
            "MT": np.array([1.51202612e-04, 6.79063452e-05, 4.09035928e-05]),
            "Total": np.array([5.66353091e-05, -2.28075165e-05, -4.67875570e-05]),
        }

        result_R, result_results = fscope_c(Ts, Tc, tau, tauphi0, R0, alpha, tau_SO)

        np.testing.assert_array_almost_equal(result_R, expected_R, decimal=5)
        for key, value in expected_results.items():
            np.testing.assert_array_almost_equal(result_results[key], value, decimal=5)

    def test_fscope_R_with_tau_SO(self) -> None:
        """Test fscope_c with tau_SO."""
        Ts = np.array([1.5, 2.0, 2.5])
        Tc = 1.0
        tau = 1e-12
        delta0 = 1e-3
        tauphi0 = pi*hbar/(8*k_B*delta0)
        R0 = 1000.0
        alpha = -1
        tau_SO = 1e-15
        expected_R = np.array([752.28932942, 803.52802669, 822.0952287])
        expected_results = {
            "AL": np.array([1.31040170e-06, 1.43936402e-07, 2.99469748e-08]),
            "MTsum": np.array([-4.04100310e-06, -1.65381459e-06, -8.44054498e-07]),
            "MTint": np.array([1.55243615e-04, 6.95601598e-05, 4.17476473e-05]),
            "DOS": np.array([-2.20475627e-06, -6.91236851e-07, -2.96615333e-07]),
            "DCR": np.array([5.54148708e-08, 1.42590054e-08, 4.65064346e-09]),
            "Fluctuation_tot": np.array([1.50363672e-04, 6.73733038e-05, 4.06415750e-05]),
            "WL": np.array([-9.37283634e-05, -9.01808203e-05, -8.74291320e-05]),
            "WAL": np.array([0.00027264, 0.00026732, 0.00026319]),
            "MT": np.array([1.51202612e-04, 6.79063452e-05, 4.09035928e-05]),
            "Total": np.array([0.0003292758061, 0.0002445116666, 0.00021640409570]),
        }

        result_R, result_results = fscope_c(Ts, Tc, tau, tauphi0, R0, alpha, tau_SO)

        np.testing.assert_array_almost_equal(result_R, expected_R, decimal=5)
        for key, value in expected_results.items():
            np.testing.assert_array_almost_equal(result_results[key], value, decimal=5)

class TestRustVsC(unittest.TestCase):
    """Compare results from the rust and C implementations of fluctuation calculation."""

    def test_mc_sigma_rust_c(self) -> None:
        """Check the two implementations give the same results."""
        t = np.array([1.1, 1.2, 1.3])
        h = np.array([0.01]*3)
        tau = np.array([0.001]*3)
        tauphi = np.array([0.001]*3)
        res1 = mc_sigma(t, h, tau, tauphi)  # returns np.array
        res2 = mc_sigma_rust(t, h, tau, tauphi)  # returns dict
        np.testing.assert_array_almost_equal(res1[0], res2["al"], decimal=6)
        np.testing.assert_array_almost_equal(res1[1], res2["mtsum"], decimal=6)
        np.testing.assert_array_almost_equal(res1[2], res2["mtint"], decimal=6)
        np.testing.assert_array_almost_equal(res1[3], res2["dos"], decimal=6)
        np.testing.assert_array_almost_equal(res1[4], res2["dcr"], decimal=6)

    def test_fscope_rust_c(self) -> None:
        """Check the two implementations give the same results."""
        Ts = np.array([1.5, 2.0, 2.5])
        Tc = 1.0
        tau = 1e-12
        tauphi0 = pi*hbar/(8*k_B*1e-3)
        R0 = 1000.0
        alpha = -1.0
        tau_SO = 1e-15
        R_rs, res_rs = fscope(Ts,Tc,tau,tauphi0,R0,alpha,tau_SO)
        R_c, res_c = fscope_c(Ts,Tc,tau,tauphi0,R0,alpha,tau_SO)
        np.testing.assert_array_almost_equal(R_rs, R_c, decimal=5)
        for key, value in res_rs.items():
            np.testing.assert_array_almost_equal(value, res_c[key], decimal=5)


if __name__ == "__main__":
    unittest.main()
