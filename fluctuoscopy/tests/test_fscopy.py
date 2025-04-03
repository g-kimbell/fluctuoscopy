"""Tests for fluctuoscopy.fluctuosco.py module."""

import unittest

import numpy as np
import pytest

from fluctuoscopy.fluctuosco import (
    _fscope_executable,
    fluc_dimless,
    fscope,
    fscope_full,
    hc2,
    simplified_al,
    weak_antilocalization,
    weak_localization,
)

e = 1.60217662e-19
m_e = 9.10938356e-31
hbar = 1.0545718e-34
k_B = 1.38064852e-23  # noqa: N816
pi = np.pi


@pytest.mark.legacy
class TestFscopeFullFunc(unittest.TestCase):
    """Tests for fscope_executable function."""

    def test_fscope_executable_empty(self) -> None:
        """Test fscope_executable with empty parameters."""
        with pytest.raises(ValueError, match="No computation type specified.*"):
            _fscope_executable({})

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
        output = _fscope_executable(params)
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

@pytest.mark.legacy
class TestFscope(unittest.TestCase):
    """Tests for fscope function."""

    def test_fscope_empty(self) -> None:
        """Test fscope with empty parameters."""
        with pytest.raises(ValueError, match="No computation type specified.*"):
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
        expected = {  # ignore header
            "t": np.array([2.0]),
            "h": np.array([0.01]),
            "SC": np.array([1.0]),
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
        expected = -(e**2) / (2 * np.pi**2 * hbar) * np.log(tau_phi / tau)
        result = weak_localization(tau, tau_phi)
        np.testing.assert_array_almost_equal(result, expected, decimal=6)


class TestWeakAntilocalization(unittest.TestCase):
    """Tests for weak_antilocalization function."""

    def test_weak_antilocalization_basic(self) -> None:
        """Test weak_antilocalization with one set of parameters."""
        tau_SO = 1e-12
        tau_phi = np.array([1e-11, 2e-11, 3e-11])
        expected = e**2 / (2 * np.pi**2 * hbar) * np.log((1 + tau_phi / tau_SO) * (1 + 2 * tau_phi / tau_SO) ** 0.5)
        result = weak_antilocalization(tau_SO, tau_phi)
        np.testing.assert_array_almost_equal(result, expected, decimal=6)


class TestSimplifiedAL(unittest.TestCase):
    """Tests for simplified_al function."""

    def test_al2d_basic(self) -> None:
        """Test simplified_al with basic parameters."""
        Ts = np.array([1.5, 2.0, 2.5])
        Tc = 1.0
        R0 = 10.0
        expected = 1 / (1 / R0 + e**2 / (16 * hbar) / np.log(Ts / Tc)) * np.heaviside(Ts - Tc, 0)
        result = simplified_al(Ts, Tc, R0)
        np.testing.assert_array_almost_equal(result, expected, decimal=6)

    def test_al2d_with_custom_c(self) -> None:
        """Test simplified_al with custom C."""
        Ts = np.array([1.5, 2.0, 2.5])
        Tc = 1.0
        R0 = 10.0
        C = 1.0
        expected = 1 / (1 / R0 + C / np.log(Ts / Tc)) * np.heaviside(Ts - Tc, 0)
        result = simplified_al(Ts, Tc, R0, C)
        np.testing.assert_array_almost_equal(result, expected, decimal=6)

    def test_al2d_below_tc(self) -> None:
        """Test simplified_al with Ts below Tc."""
        Ts = np.array([1.0, 1.5, 1.9])
        Tc = 2.0
        R0 = 10.0
        expected = np.zeros_like(Ts)
        result = simplified_al(Ts, Tc, R0)
        np.testing.assert_array_almost_equal(result, expected, decimal=6)


class TestFlucDimless(unittest.TestCase):
    """Tests for fluc_dimless function."""

    def test_fluc_dimless_basic(self) -> None:
        """Test fluc_dimless with basic parameters."""
        t = np.array([1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0])
        h = np.array([0.01] * 10)
        Tctau = np.array([0.01] * 10)
        Tctauphi = np.array([0.01] * 10)

        result = fluc_dimless(t, h, Tctau, Tctauphi)

        # expected is row-wise t, h, SC, sAL, sMTsum, sMTint, sDOS, sDCR, sigma
        expected = np.array(
            [
                [1.1, 0.01, 1, 5.022763e-01, -2.127955e-01, 5.515951e-03, -1.188322e-01, 2.201079e-02, 1.981753e-01],
                [1.2, 0.01, 1, 2.135999e-01, -1.581277e-01, 3.970320e-03, -7.713061e-02, 1.675453e-02, -9.335331e-04],
                [1.3, 0.01, 1, 1.228730e-01, -1.274563e-01, 3.185333e-03, -5.679587e-02, 1.316006e-02, -4.503382e-02],
                [1.4, 0.01, 1, 7.980829e-02, -1.058978e-01, 2.673269e-03, -4.426497e-02, 1.037114e-02, -5.731012e-02],
                [1.5, 0.01, 1, 5.566635e-02, -9.045830e-02, 2.314810e-03, -3.583477e-02, 8.395010e-03, -5.991690e-02],
                [1.6, 0.01, 1, 4.037312e-02, -7.809604e-02, 2.035380e-03, -2.964498e-02, 6.801725e-03, -5.853079e-02],
                [1.7, 0.01, 1, 2.997796e-02, -6.783675e-02, 1.805853e-03, -2.487227e-02, 5.500933e-03, -5.542427e-02],
                [1.8, 0.01, 1, 2.282688e-02, -5.972182e-02, 1.622634e-03, -2.119898e-02, 4.530199e-03, -5.194108e-02],
                [1.9, 0.01, 1, 1.758811e-02, -5.276137e-02, 1.464577e-03, -1.821242e-02, 3.727568e-03, -4.819353e-02],
                [2.0, 0.01, 1, 1.365203e-02, -4.670521e-02, 1.325284e-03, -1.573297e-02, 3.060543e-03, -4.440032e-02],
            ],
        )
        np.testing.assert_array_almost_equal(result["sc"], ~expected[:, 2].astype(bool), decimal=6)
        np.testing.assert_array_almost_equal(result["al"], expected[:, 3], decimal=6)
        np.testing.assert_array_almost_equal(result["mtsum"], expected[:, 4], decimal=6)
        np.testing.assert_array_almost_equal(result["mtint"], expected[:, 5], decimal=6)
        np.testing.assert_array_almost_equal(result["dos"], expected[:, 6], decimal=6)
        np.testing.assert_array_almost_equal(result["dcr"], expected[:, 7], decimal=6)


class TestFscopeFluc(unittest.TestCase):
    """Tests for fscope function."""

    def test_fscope_c(self) -> None:
        """Test fscope with basic parameters."""
        Ts = np.array([1.5, 2.0, 2.5])
        Tc = 1.0
        tau = 1e-12
        delta0 = 1e-3
        tauphi0 = pi * hbar / (8 * k_B * delta0)
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
            "WAL": np.array([0.0, 0.0, 0.0]),
            "MT": np.array([1.51202612e-04, 6.79063452e-05, 4.09035928e-05]),
            "Total": np.array([5.66353091e-05, -2.28075165e-05, -4.67875570e-05]),
        }

        result_R, result_results = fscope(Ts, Tc, tau, tauphi0, R0, alpha, tau_SO)

        np.testing.assert_array_almost_equal(result_R, expected_R, decimal=5)
        for key, value in expected_results.items():
            np.testing.assert_array_almost_equal(result_results[key], value, decimal=5)

    def test_fscope_with_tau_so(self) -> None:
        """Test fscope with tau_SO."""
        Ts = np.array([1.5, 2.0, 2.5])
        Tc = 1.0
        tau = 1e-12
        delta0 = 1e-3
        tauphi0 = pi * hbar / (8 * k_B * delta0)
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

        result_R, result_results = fscope(Ts, Tc, tau, tauphi0, R0, alpha, tau_SO)

        np.testing.assert_array_almost_equal(result_R, expected_R, decimal=5)
        for key, value in expected_results.items():
            np.testing.assert_array_almost_equal(result_results[key], value, decimal=5)


class TestHC2(unittest.TestCase):
    """Check Hc2 rust port gives expected results."""

    def test_hc2(self) -> None:
        """Compare Hc2 rust to reuslts from C++ implementation."""
        t = np.linspace(0, 1.1, 12)
        expect = np.array(
            [
                0.6926728737556,
                0.6787010770604,
                0.6417110293611,
                0.5886332291516,
                0.5239696917854,
                0.4505228358223,
                0.3701313737292,
                0.2840686407473,
                0.1932568936088,
                0.0983887684337,
                0.0,
                0.0,
            ],
        )
        np.testing.assert_array_almost_equal(hc2(t), expect, decimal=6)


if __name__ == "__main__":
    unittest.main()
