<p align="center">
  <img src="https://github.com/user-attachments/assets/8bf0686c-e963-4169-ada5-9598d3949553" width="500" align="center" alt="fluctuoscopy">
</p>

A python wrapper for the C++ [FSCOPE](https://github.com/andreasglatz/FSCOPE) program written by [Andreas Glatz](https://github.com/andreasglatz), for calculating conductivity contributions of superconducting fluctuations.

## Installation

Clone the repo, navigate to its directory and use
```
pip install .
```
To include testing use:
```
pip install .[dev]
pytest
```

## Usage

To see possible parameters use
```
import fluctuoscopy
fluctuoscopy.fscope()
```
The fscope function accepts a dictionary e.g.
```
fluctuoscopy.fscope({'ctype':100,'hmin':0.01})
```
this will return a dictionary of calculated fluctuation contributions.

## Contributors

- [Graham Kimbell](https://github.com/g-kimbell)
- [Ulderico Filippozzi](https://github.com/ufilippozzi)
- [Andreas Glatz](https://github.com/andreasglatz)
