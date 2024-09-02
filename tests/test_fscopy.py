import unittest
import numpy as np
from fluctuoscopy.fluctuosco import AL2D, calc_tau, weak_localization, weak_antilocalization, fscope_full_func, fscope_delta_wrapped

e = 1.60217662e-19
m_e = 9.10938356e-31
hbar = 1.0545718e-34
pi = np.pi

class TestFscopeFullFunc(unittest.TestCase):
    def test_fscope_full_func_empty(self):
        result = fscope_full_func({})
        self.assertIsInstance(result,list)
        self.assertGreater(len(result),0)
        self.assertIn('FLUCTUOSCOPE',result[0])

    def test_fscope_full_func_basic(self):
        params = [
            'ctype=100',
            'tmin=2',
            'dt=0',
            'Nt=1',
            'hmin=0.01',
            'dh=0',
            'Nh=1',
            'Tc0tau=1e-2',
            'delta=1e-3'
        ]
        output = fscope_full_func(params)
        result = [float(r) for r in output[-1].split('\t')]
        expected = [
            2.0, 0.01, 1.0, 0.013652032863547285, -0.04670521178360401, 0.4186962142931441,
            -0.015732975271415026, 0.0030605431719476525, 0.37297060327361997
        ]
        np.testing.assert_array_almost_equal(result, expected, decimal=6)

class TestFscopeDeltaWrapped(unittest.TestCase):
    def test_fscope_delta_wrapped_basic(self):
        Ts = np.array([1.5, 2.0, 2.5])
        Tc = 1.0
        tau = 1e-12
        delta0 = 1e-3
        R0 = 1000.0
        alpha = -1
        tau_SO = None

        expected_R = [946.40032506, 1023.33984024, 1049.08408124]
        expected_results = {
            'SC': np.array([1., 1., 1.]),
            'AL': np.array([1.31040170e-06, 1.43936402e-07, 2.99469748e-08]),
            'MTsum': np.array([-4.04100310e-06, -1.65381459e-06, -8.44054498e-07]),
            'MTint': np.array([1.55243615e-04, 6.95601598e-05, 4.17476473e-05]),
            'DOS': np.array([-2.20475627e-06, -6.91236851e-07, -2.96615333e-07]),
            'DCR': np.array([5.54148708e-08, 1.42590054e-08, 4.65064346e-09]),
            'Fluctuation_tot': np.array([1.50363672e-04, 6.73733038e-05, 4.06415750e-05]),
            'WL': np.array([-9.37283634e-05, -9.01808203e-05, -8.74291320e-05]),
            'WAL': np.array([0., 0., 0.]), 'MT': np.array([1.51202612e-04, 6.79063452e-05, 4.09035928e-05]),
            'Total': np.array([5.66353091e-05, -2.28075165e-05, -4.67875570e-05])
        }

        result_R, result_results = fscope_delta_wrapped(Ts, Tc, tau, delta0, R0, alpha, tau_SO)

        np.testing.assert_array_almost_equal(result_R, expected_R, decimal=6)
        for key in expected_results:
            np.testing.assert_array_almost_equal(result_results[key], expected_results[key], decimal=6)

    def test_fscope_delta_wrapped_with_tau_SO(self):
        Ts = np.array([1.5, 2.0, 2.5])
        Tc = 1.0
        tau = 1e-12
        delta0 = 1e-3
        R0 = 1000.0
        alpha = -1
        tau_SO = 1e-15
        expected_R = np.array([752.28932942, 803.52802669, 822.0952287])
        expected_results = {
            'SC': np.array([1., 1., 1.]),
            'AL': np.array([1.31040170e-06, 1.43936402e-07, 2.99469748e-08]),
            'MTsum': np.array([-4.04100310e-06, -1.65381459e-06, -8.44054498e-07]),
            'MTint': np.array([1.55243615e-04, 6.95601598e-05, 4.17476473e-05]),
            'DOS': np.array([-2.20475627e-06, -6.91236851e-07, -2.96615333e-07]),
            'DCR': np.array([5.54148708e-08, 1.42590054e-08, 4.65064346e-09]),
            'Fluctuation_tot': np.array([1.50363672e-04, 6.73733038e-05, 4.06415750e-05]),
            'WL': np.array([-9.37283634e-05, -9.01808203e-05, -8.74291320e-05]),
            'WAL': np.array([0.00027264, 0.00026732, 0.00026319]),
            'MT': np.array([1.51202612e-04, 6.79063452e-05, 4.09035928e-05]),
            'Total': np.array([5.66353091e-05, -2.28075165e-05, -4.67875570e-05])
        }

        result_R, result_results = fscope_delta_wrapped(Ts, Tc, tau, delta0, R0, alpha, tau_SO)

        np.testing.assert_array_almost_equal(result_R, expected_R, decimal=6)
        for key in expected_results:
            np.testing.assert_array_almost_equal(result_results[key], expected_results[key], decimal=6)

class TestWeakLocalization(unittest.TestCase):
    def test_weak_localization_basic(self):
        tau = 1e-12
        tau_phi = np.array([1e-11, 2e-11, 3e-11])
        expected = -e**2 / (2 * np.pi**2 * hbar) * np.log(tau_phi / tau)
        result = weak_localization(tau, tau_phi)
        np.testing.assert_array_almost_equal(result, expected, decimal=6)

class TestWeakAntilocalization(unittest.TestCase):
    def test_weak_antilocalization_basic(self):
        tau_SO = 1e-12
        tau_phi = np.array([1e-11, 2e-11, 3e-11])
        expected = e**2 / (2 * np.pi**2 * hbar) * np.log(
            (1 + tau_phi / tau_SO) * (1 + 2 * tau_phi / tau_SO)**0.5
        )
        result = weak_antilocalization(tau_SO, tau_phi)
        np.testing.assert_array_almost_equal(result, expected, decimal=6)

class TestCalcTau(unittest.TestCase):
    def test_calc_tau_basic(self):
        rel_eff_mass = 0.1
        RN = 100.0
        Vg = 1.0
        Vg_n = np.array([0.5, 1.0, 1.5])
        n = np.array([1e15, 2e15, 3e15])
        expected = rel_eff_mass * m_e / (np.interp(Vg, Vg_n, n) * RN * e**2)
        result = calc_tau(rel_eff_mass, RN, Vg, Vg_n, n)
        self.assertAlmostEqual(result, expected, places=6)

class TestAL2D(unittest.TestCase):
    def test_al2d_basic(self):
        Ts = np.array([1.5, 2.0, 2.5])
        Tc = 1.0
        R0 = 10.0
        expected = 1 / (1 / R0 +  e**2 / (16*hbar) / np.log(Ts / Tc)) * np.heaviside(Ts - Tc, 0)
        result = AL2D(Ts, Tc, R0)
        np.testing.assert_array_almost_equal(result, expected, decimal=6)

    def test_al2d_with_custom_C(self):
        Ts = np.array([1.5, 2.0, 2.5])
        Tc = 1.0
        R0 = 10.0
        C = 1.0
        expected = 1 / (1 / R0 + C / np.log(Ts / Tc)) * np.heaviside(Ts - Tc, 0)
        result = AL2D(Ts, Tc, R0, C)
        np.testing.assert_array_almost_equal(result, expected, decimal=6)

    def test_al2d_below_tc(self):
        Ts = np.array([1.0, 1.5, 1.9])
        Tc = 2.0
        R0 = 10.0
        expected = np.zeros_like(Ts)
        result = AL2D(Ts, Tc, R0)
        np.testing.assert_array_almost_equal(result, expected, decimal=6)

if __name__ == '__main__':
    unittest.main()
