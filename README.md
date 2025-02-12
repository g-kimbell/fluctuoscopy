<p align="center">
  <img src="https://github.com/user-attachments/assets/d2d4424c-d948-4aca-ab86-12e0b3c4638e#gh-light-mode-only" width="500" align="center" alt="fluctuoscopy">
  <img src="https://github.com/user-attachments/assets/16b4d081-529d-4018-b4d8-a9f12614231c#gh-dark-mode-only" width="500" align="center" alt="fluctuoscopy">
</p>

A Python wrapper for the C++ [FSCOPE](https://github.com/andreasglatz/FSCOPE) program written by [Andreas Glatz](https://github.com/andreasglatz), for calculating conductivity contributions of superconducting fluctuations.

## Installation

Clone the repo, navigate to its directory and use
```
pip install git+https://github.com/g-kimbell/fluctuoscopy.git
```

## Usage

To see possible parameters use
```
import fluctuoscopy
fluctuoscopy.fscope({})
```
The fscope function accepts a dictionary e.g.
```
fluctuoscopy.fscope({'ctype':100,'hmin':0.01})
```
This will return a dictionary of calculated fluctuation contributions.

## Development and testing

For development/testing, go to a folder and clone the repo
```
git clone https://github.com/g-kimbell/fluctuoscopy
```
Then go inside the repo and install in editable mode
```
cd fluctuoscopy
pip install -e .
```
You can then make changes. For testing install and run pytest
```
pip install pytest
pytest
```

## Contributors

- [Graham Kimbell](https://github.com/g-kimbell)
- [Ulderico Filippozzi](https://github.com/ufilippozzi)
- [Andreas Glatz](https://github.com/andreasglatz)
