# feslib
Thermodinamics, phase transitions, appications for different fluids, equations of state."

## Description

**feslib** provides several functions and example scripts for investigation of phase transitions.
Several different equations of state are being investigated.
The library is based on the results of articles by Valentin Lychagin, his students and colleagues.

Useful bibliography:

 [Valentin Lychagin Thermodynamics as a theory of measurement](https://doi.org/10.1016/j.geomphys.2021.104430)



#### Contribute
This code is still under development and benchmarking. If you find any bugs or errors in the code, please report them in GitHub.

For this code, we work on the develop branch and merge it to the main branch (with a new version number) everytime significant addtions/improvements are made. If you plan on making contributions, please base everything on the develop branch.

### Benchmarks
---

## Methods
`Beattie-Bridgeman` Phase transitions using Beattie-Bridgeman EOS.
 [Beattie J.A., Bridgeman O.C. An Equation of State for Gaseous Mixtures. 1. Application to Mixtures of Methane and Nitrogen // J. Am. Chem. Soc. 1929. V. 51. P. 19–30.](https://doi.org/10.1021/ja01376a003)

...

## Example scripts
`Run_Beattie-Bridgeman` Beattie-Bridgeman example.

## How to install and run
If you would like to modify the source code, download the feslib repository and install using pip (or pip3 depending on your installation).
```bash
    git clone https://github.com/LychaginTeam/feslib.git
    cd feslib/
    pip install .
```
Alternatively, you can install Displacement-strain-planet via pip
```bash
   pip install feslib
```

## To run the example scripts
```bash
    cd examples
    jupyter notebook Run_demo.ipynb
    python feslib_example.py
```

## Authors
[Maksim Kostiuchek](https://www.ipu.ru/node/47150) (max31@list.ru),
[Alexey Batov](https://www.ipu.ru/node/82) (batov@ipu.ru),
[Anton Salnikov](https://www.ipu.ru/staff/salnikov) (salnikov@ipu.ru)
[Ivan Galyaev](https://www.ipu.ru/node/49970) (ivan.galyaev@yandex.ru),
[Valentin Lychagin](https://www.ipu.ru/node/457)

## Cite
You can cite the latest release of the package as:
feslib: 0.1.0 (Version 0.1.0). Zenodo. http://doi.org/...

## Acknowledgments
The development of this library was supported by Russian Science Foundation grant number 21-71-20034.
