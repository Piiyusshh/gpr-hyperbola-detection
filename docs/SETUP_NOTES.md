# Setup notes

## Checked environment

- OS reported by PowerShell: Windows 10 Home Single Language, version 2009.
- `python --version`: Python 3.12.2.
- `gcc` and `gfortran` were found at `C:\MinGW\bin`. `cl` (Microsoft's C
  compiler) was not found. A compiler was not required because pip selected
  the official Windows binary wheel.

## Installation record

I followed the official gprMax PyPI/virtual-environment route:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install gprMax matplotlib
```

The initial sandboxed pip attempt failed with Windows socket error 10013. The
download was then allowed and pip selected `gprMax-4.0.1-cp312-cp312-win_amd64.whl`.
The combined install did not complete in one command window, so I installed
the gprMax wheel and its dependencies in smaller commands and checked imports.

`python -m gprMax --help` ran successfully. The installed official example
`cylinder_Ascan_2D.in` initially failed because it asks gprMax to write a
geometry view to `results/`, which did not exist. Creating `results/` fixed
that file-location error; the rerun completed and wrote
`results/official_example.h5`.
