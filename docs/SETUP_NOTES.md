# Setup notes

## Recommended environment

Use a Python virtual environment. The project was tested with Python 3.12 on
Windows and gprMax 4.0.1. Other operating systems or Python versions may
require separate validation.

## Installation record

The tested installation followed the official gprMax
PyPI/virtual-environment route:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install gprMax matplotlib
```

The repository's `requirements.txt` records the tested package environment.
Installation may take longer on a fresh machine because gprMax and scientific
Python packages have compiled dependencies.

`python -m gprMax --help` ran successfully. The installed official example
`cylinder_Ascan_2D.in` initially failed because it asks gprMax to write a
geometry view to `results/`, which did not exist. Creating `results/` fixed
that file-location error; the rerun completed and wrote
the local `results/official_example.h5` output.

## Historical local-machine note

The original setup investigation was performed on Windows 10 with Python
3.12.2. It also recorded local compiler discovery, including a MinGW
installation. Those details are historical observations from one machine,
not project requirements; a compiler was not required for the tested Windows
wheel installation.
