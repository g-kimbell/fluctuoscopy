use num_complex::*;
use polygamma::polygamma;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use pyo3::PyObject;
use pyo3::PyResult;
use rayon::prelude::*;
use spfunc::gamma::digamma as complex_digamma;
use statrs::function::gamma::digamma;
use std::vec::Vec;


#[pyfunction]
pub fn mc_sigma(py: Python, t: f64, h: f64, tctau: f64, tctauphi: f64) -> PyResult<PyObject> {
    let (al, mtsum, mtint, dos, dcr, sc) = calculate_mc_sigma(t, h, tctau, tctauphi);

    let dict = PyDict::new(py);
    dict.set_item("al", al)?;
    dict.set_item("mtsum", mtsum)?;
    dict.set_item("mtint", mtint)?;
    dict.set_item("dos", dos)?;
    dict.set_item("dcr", dcr)?;
    dict.set_item("sc", sc)?;

    Ok(dict.into())
}

#[pyfunction]
pub fn mc_sigma_parallel(
    py: Python,
    t_values: Vec<f64>,
    h_values: Vec<f64>,
    tctau_values: Vec<f64>,
    tctauphi_values: Vec<f64>,
) -> PyResult<PyObject> {
    // Check that all input arrays have the same length
    let n = t_values.len();
    assert_eq!(
        h_values.len(),
        n,
        "All input arrays must have the same length"
    );
    assert_eq!(
        tctau_values.len(),
        n,
        "All input arrays must have the same length"
    );
    assert_eq!(
        tctauphi_values.len(),
        n,
        "All input arrays must have the same length"
    );

    // Pre-allocate result vectors
    let mut al = Vec::with_capacity(n);
    let mut mtsum = Vec::with_capacity(n);
    let mut mtint = Vec::with_capacity(n);
    let mut dos = Vec::with_capacity(n);
    let mut dcr = Vec::with_capacity(n);
    let mut sc = Vec::with_capacity(n);

    // Create input parameter tuples
    let params: Vec<(f64, f64, f64, f64)> = t_values
        .into_iter()
        .zip(h_values.into_iter())
        .zip(tctau_values.into_iter())
        .zip(tctauphi_values.into_iter())
        .map(|(((t, h), tctau), tctauphi)| (t, h, tctau, tctauphi))
        .collect();

    // Process in parallel and collect results
    let results: Vec<(f64, f64, f64, f64, f64, bool)> = params
        .into_par_iter()
        .map(|(t, h, tctau, tctauphi)| calculate_mc_sigma(t, h, tctau, tctauphi))
        .collect();

    // Extract values into separate vectors
    for (s_al, s_mtsum, s_mtint, s_dos, s_dcr, s_sc) in results {
        al.push(s_al);
        mtsum.push(s_mtsum);
        mtint.push(s_mtint);
        dos.push(s_dos);
        dcr.push(s_dcr);
        sc.push(s_sc);
    }

    let dict = PyDict::new(py);
    dict.set_item("al", al)?;
    dict.set_item("mtsum", mtsum)?;
    dict.set_item("mtint", mtint)?;
    dict.set_item("dos", dos)?;
    dict.set_item("dcr", dcr)?;
    dict.set_item("sc", sc)?;

    Ok(dict.into())
}

#[pyfunction]
pub fn hc2_parallel(_py: Python, t_values: Vec<f64>) -> PyResult<Vec<f64>> {
    let hc2_values: Vec<f64> = t_values.par_iter().map(|&t| hc2(t)).collect();
    Ok(hc2_values)
}

fn calculate_mc_sigma(t: f64, mut h: f64, tctau: f64, tctauphi: f64) -> (f64, f64, f64, f64, f64, bool) {
    // Calculate the fluctuation conductivity components

    // Check if already superconducting
    if (t<=1.0) && ((h==0.0) || (h > hc2(t))) {
        return (0.0, 0.0, 0.0, 0.0, 0.0, true);
    }

    let kz: i32 = 200;
    let z: f64 = 5.0;
    let gphi: f64 = 0.125 * PI / (t * tctauphi);

    let mut s_al = 0.0;
    let mut s_mtint = 0.0;
    let mut s_mtsum = 0.0;
    let mut s_dos = 0.0;
    let mut s_cc = 0.0;

    if h < 0.0001 {
        h = 0.0001;
    }

    let m_max: i32 = (1.0 / (t * tctau) + 0.5) as i32;
    let mut res = 0;
    for m in (0..=m_max).rev() {
        let rs = mc_ksum(m, t, h);
        let rs_thing = rs.0;
        let sum_mt = rs.1;
        let sum_cc = rs.2;
        if rs_thing > res {
            res = rs_thing;
        }
        let (i_al, i_mt, i_dos, _i_cc) = mc_int(m, t, h, z, kz);
        s_al += (1.0 + m as f64) * i_al;
        s_mtint += C_MTI / (gphi + 2.0 * h / t * (m as f64 + 0.5)) * i_mt;
        s_mtsum += sum_mt;
        s_dos += i_dos;
        s_cc += (m as f64 + 0.5) * sum_cc;
    }
    let sig_al = s_al / PI;
    let sig_mtint = C_MT * s_mtint * h / t;
    let sig_mtsum = C_MT * s_mtsum * h / t;
    let sig_dos = C_DOS * s_dos * h / t;
    let sig_cc = C_CC * s_cc * h * h / (t * t);
    (sig_al, sig_mtsum, sig_mtint, sig_dos, sig_cc, false)
}

fn mc_ksum(n: i32, t: f64, h: f64) -> (i32, f64, f64) {
    let mut mt = 0.0;
    let mut cc = 0.0;

    let mut res = 0;
    let mut t_calc = t;
    if t < 1e-6 {
        t_calc = 1e-6;
        res = 2;
    }

    let xn = C1 * (n as f64 + 0.5) * h / t_calc;

    let mut km = 2000 - (2.0 * xn) as i32;
    if km < 2 {
        km = 2;
    }

    // Now do the integration
    let am = (h * (n as f64 + 0.5)).ln() - C2 - C6;
    let zmax = 1.0 / (2.0 * C1 + t_calc * km as f64 / (h * (n as f64 + 0.5)));

    let nz = 25; // Functions are smooth enough and zmax < 1.2
    let dz = zmax / nz as f64;
    let dzh = 0.5 * dz;

    for k in 0..nz {
        for j in 0..5 {
            let z = k as f64 * dz + dzh * (1.0 + GLX[j]);
            let x = 1.0 / (am - z.ln());
            cc += GLW[j] * x * z;
            mt += GLW[j] * x;
        }
    }

    let x = -t_calc / (h * (n as f64 + 0.5));
    cc = x * x * dzh * cc;
    mt = x * dzh * mt;

    if km > 2 {
        // Need summation
        // Shift k by one, then sum from 2 to kmax
        let am = t_calc.ln() - C2;
        for k in (2..=km - 1).rev() {
            let x = 0.5 * k as f64 + xn;
            let en = am + digamma(x);
            let en2 = polygamma(2, x).expect("oh no polygamma2 broke");
            let en3 = polygamma(3, x).expect("oh no polygamma3 broke");
            mt += en2 / en;
            cc += en3 / en;
        }
    }

    // k=0 term
    let x = 0.5 + xn;
    let en = t_calc.ln() - C2 + digamma(x);
    let en2 = polygamma(2, x).expect("oh no polygamma2 broke");
    let en3 = polygamma(3, x).expect("oh no polygamma3 broke");
    mt = 2.0 * mt + en2 / en;
    cc = 2.0 * cc + en3 / en;

    (res, mt, cc)
}

fn mc_int(n: i32, t: f64, h: f64, zmax: f64, zsteps: i32) -> (f64, f64, f64, f64) {

    let dz = zmax / zsteps as f64;
    let dzh = 0.5 * dz;

    let mut s_al = 0.0;
    let mut s_mt = 0.0;
    let mut s_dos = 0.0;

    for i in -zsteps..zsteps {
        for j in 0..5 {
            let z = i as f64 * dz + dzh * (1.0 + GLX[j]);
            let (al, mt, dos, _cc) = mc_func(n, t, h, z);
            s_al += GLW[j] * al;
            s_mt += GLW[j] * mt;
            s_dos += GLW[j] * dos;
            // s_cc += GLw[j] * cc; // This line was commented out in the original...
        }
    }

    let al = dzh * s_al;
    let mt = 2.0 * dzh * s_mt;
    let dos = dzh * s_dos;
    let cc = 0.0; // In the original: CC=0; // 4*dzh*s_cc;
    (al, mt, dos, cc)
}

fn mc_func(n: i32, t: f64, h: f64, mut z: f64) -> (f64, f64, f64, f64) {
    if z.abs() < INT0 {
        if z >= 0.0 {
            z = INT0;
        } else {
            z = -INT0;
        }
    }

    let (enr, eni) = e_n(n, t, h, z);
    let (en1r, en1i) = e_n(n + 1, t, h, z);

    // Bad section as commented in the original

    let psip: f64 = complex_digamma(Complex64::new(
        0.5 + C1 * h / t * (n as f64 + 0.5) + DX,
        0.5 * z,
    ))
    .im();
    let psim: f64 = complex_digamma(Complex64::new(
        0.5 + C1 * h / t * (n as f64 + 0.5) - DX,
        0.5 * z,
    ))
    .im();

    let imen1 = 0.25 * (psip - psim) / DX; // Im E_n'

    // Denominator
    let absn = enr * enr + eni * eni;
    let absn1 = en1r * en1r + en1i * en1i;
    let d = (PI * z).sinh();
    let d = 1.0 / (d * d * absn);

    // Nominator
    let dr = enr - en1r;
    let di = eni - en1i;
    let res = (dr * dr - di * di) * eni * en1i - dr * di * (eni * en1r + en1i * enr);

    let al = res * d / absn1;
    let mt = eni * eni * d;
    let dos = eni * imen1 * d;
    let cc = 0.0; // eni * imen2 * d; (commented out in original)

    (al, mt, dos, cc)
}

fn hc2(t: f64) -> f64 {
    //Caculate hc2 for a given t.
    if t >= 1.0 {
        return 0.0;
    }
    if t <= 0.0 {
        return HC20;
    }
    //Binary search solution of log(t)+Psi(1/2+2/pi^2*h/t)-Psi(1/2)=0
    let c = t.log(E) - C2;
    let mut h_low = (1.0 - t) * HC20;
    let mut h_high = HC20;
    let mut h_mid = 0.5 * (h_low + h_high);
    for _ in 0..32 { // <10^-9 error
        let f = c + digamma(0.5 + C5 * h_mid / t);
        if f < 0.0 {
            h_low = h_mid;
        } else {
            h_high = h_mid;
        }
        h_mid = 0.5 * (h_low + h_high);
    }
    h_mid
}

// Helper functions needed by mc_func
fn e_n(n: i32, t: f64, h: f64, z: f64) -> (f64, f64) {
    let compthing: Complex64 = t.log(E) - C2
        + complex_digamma(Complex64::new(0.5 + C1 * h / t * (n as f64 + 0.5), 0.5 * z));
    (compthing.re(), compthing.im())
}

const PI: f64 = std::f64::consts::PI;
const E: f64 = std::f64::consts::E;

//coordiantes & weights for Gauss-Legendre 5 point integration
const GLX: [f64; 5] = [
    -0.90617984593866399280,
    -0.53846931010568309104,
    0.0,
    0.53846931010568309104,
    0.90617984593866399280,
];
const GLW: [f64; 5] = [
    0.23692688505618908751,
    0.47862867049936646804,
    0.56888888888888888889,
    0.47862867049936646804,
    0.23692688505618908751,
];
const C1: f64 = 0.405284734569351085775517852838911; //4/pi^2 
const C2: f64 = -1.96351002602142347944097633299876; //Psi(1/2)
const C5: f64 = 0.202642367284675542887758926419455; //2/pi^2
const C6: f64 = 0.693147180559945309417232121458177; //ln(2)

const C_MTI: f64 = 15.503138340149910087738157533551; //pi^3/2 (factor 2 is in the integral)
const C_MT: f64 = 0.010265982254684335189152783267119; //1/pi^4
const C_DOS: f64 = 0.129006137732797956737688210754255; //4/pi^3
const C_CC: f64 = 0.00138688196439446972811978351321894; //4/pi^6/3
const HC20: f64 = 0.69267287375563603674263246549077793763519897; //=pi^2*exp[-gamma_e]/8
const INT0: f64 = 10e-10; //integral cutoff at z=0
const DX: f64 = 10e-6;
