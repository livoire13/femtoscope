![](images/logo.png)

The *femtoscope* software was written as part of Hugo Lévy's PhD thesis (2021-2024), under the supervision of Joël Bergé.
This Python code can be used to investigate different models of modified gravity within complex geometries on unbounded domains.
*femtoscope* depends on many open source softwares (see dependencies), and comes as an alternative to commercial softwares.

## Installation from YAML

YAML (stands for YAML Ain't Markup Language) is a serialization
language which is often used asa format for configuration files.
Anaconda allows the creation of an environment file (with .yml
extension) in order to share with others.

>> conda env export --name femtoscope > femtoscope.yml

This will create femtoscope.yml at current location. This file can then be used by another Anaconda user to re-create the same environment. The steps to follow (from official conda online documentation) are:
1) Create the environment >> conda env create -f femtoscope.yml
2) Activate the new environment >> conda activate femtoscope
3) Verify that the new environment was installed correctly
    >> conda env list
   and make sure femtoscope appears in the list.
This process is likely to take several tens of minutes (depends on the user's Internet connection speed).
   
**note**
The project uses Python 3.9.16

## Main dependencies

- Python 3.9.16
- meshio 4.4.6
- numpy 1.25.0
- pandas 1.5.3
- pyevtk 1.5.0
- pyvista 0.38.3
- scipy 1.10.1
- sfepy 2023.2
- pdoc 10.0.1
- gmsh  4.11.1

## setup (PYTHONPATH & more)

The source directory of *femtoscope* is deliberately not placed in Anaconda3\Lib\site-packages as it is still under development. In order to be able to load the module from any location, one needs to add the *femtoscope* directory to the PYTHONPATH variable.

Useful tips:

- Display PYTHONPATH
	>> import sys
	>> print(sys.path)

- Modify PYTHONPATH
	with spyder --> Tools -> PYTHONPATH manager -> + Add path
	The selected path must point to the *femtoscope* parent folder.
	in windows system environment variable --> modify PYTHONPATH
	
## Tree structure

```shell
├───data  # I/O files
│   ├───mesh
│   │   └───geo
│   ├───model
│   └───result
│       └───plot
├───femtoscope
│   ├───core  # weak form representation & solver
│   ├───display
│   ├───inout  # mesh and .vtk generation
│   ├───misc
│   └───physics  # Poisson & KG equations
├───images  # femtoscope logo
└───script  # examples
```

## Citing

If you would like to cite *femtoscope* in a paper or presentation, please use the following reference:
```
@article{hlevy:femtoscope,
	doi = {10.1103/PhysRevD.106.124021},
	url = {https://link.aps.org/doi/10.1103/PhysRevD.106.124021},
	author = {Lévy, Hugo and Bergé, Joël and Uzan, Jean-Philippe},
	title = {Solving nonlinear Klein-Gordon equations on unbounded domains via
	the Finite Element Method},
	month = {09},
	year = {2022}
}
```

## Generate documentation

The documentation is generated using pdoc by running the Python script "generate_documentation.py" located in the "script" directory.