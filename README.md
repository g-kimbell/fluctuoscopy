<p align="center">
  <img src="https://github.com/user-attachments/assets/d2d4424c-d948-4aca-ab86-12e0b3c4638e#gh-light-mode-only" width="500" align="center" alt="fluctuoscopy">
  <img src="https://github.com/user-attachments/assets/16b4d081-529d-4018-b4d8-a9f12614231c#gh-dark-mode-only" width="500" align="center" alt="fluctuoscopy">
</p>

Calculate conductivity contributions of superconducting fluctuations.

A blazingly fast<sup>TM</sup> Rust/Python port for the C++ [FSCOPE](https://github.com/andreasglatz/FSCOPE) program written by [Andreas Glatz](https://github.com/andreasglatz).


## Installation

With Python >3.9 use
```
pip install fluctuoscopy
```


## Usage

Calculate R(T) from the fscope function in SI units:
```
import fluctuoscopy
R, sigmas = fluctuoscopy.fscope(T, Tc, tau, tau_phi0, R0, alpha, tau_SO)
```
Where the inputs are floats or numpy arrays, and the outputs are an array of resistances and a dictionary of conductivity contributions, in SI units.

You can also do dimensionless calculations, returning conductivity contributions in units G0:
```
sigmas = fluctuoscopy.fluc_dimless(t, h, Tc_tau, Tc_tauphi)
```

These calculations have optimised Rust ports. To see other possible calculations use
```
fluctuoscopy.fscope_full({})
```
Where parameters are passed in as a dictionary, see the FSCOPE documentation for a full list of calculations and parameters.

Note that this uses a pre-compiled version of FSCOPE, we compiled for Windows x86, macos x86 and arm64, and linux x86. If you are on a different platform, you will have compile FSCOPE yourself.


## Development and testing

For development/testing, clone the repo
```
git clone https://github.com/g-kimbell/fluctuoscopy
```
Then go inside the repo and install in editable mode
```
cd fluctuoscopy
pip install -e .
```
The project uses maturin, if you edit the Rust code or if you are on a particularly weird platform that we have not already compiled for, you will have to recompile the binaries then reinstall the module
```
pip install maturin
maturin build --release
pip install -e .
```
For testing install and run pytest
```
pip install pytest
pytest
```


## Contributors

- [Graham Kimbell](https://github.com/g-kimbell)
- [Ulderico Filippozzi](https://github.com/ufilippozzi)
- [Andreas Glatz](https://github.com/andreasglatz)
