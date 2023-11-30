# feslib
Thermodinamics, phase transitions, appications for different fluids, equations of state.

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
`Beattie-Bridgeman` Phase transitions using Beattie-Bridgeman EOS. The Beattie–Bridgeman model was chosen to describe phase transitions and their features for the equation of state of a real gas. The description of the method is in the following article: 
[I. A. Galyaev, M. I. Kostiuchek, A. V. Batov, and A. M. Salnikov. Critical Phenomena of Massieu–Plank Potential for Gas Mixtures Described by the Beattie–Bridgeman Equations of State // Lobachevskii Journal of Mathematics, 2023, Vol. 44, No. 9, pp. 3919–3926](https://doi.org/10.1134/S1995080223090093)

The paper considers thermodynamics as a measurement of extensive variables, such as energy, volume and mass. In this sense, thermodynamic states are Legendrian or Lagrangian surfaces in the corresponding contact or symplectic space. The Beattie-Bridgeman model was chosen to describe phase transitions and their features for the equation of state of a real gas. This model describes the state of a substance in two phases: liquid and vapor. Real gas can be either single gase or a mixture. The program provides calculated data for many gas mixtures: methane, ethane, propane, butane, pentane, hydrogen, nitrogen, carbon dioxide, ammonia. Using the program, you can calculate formulas for model constants for any mixture of gases. You can obtain graphs of: the caloric equation of state and various phase transition potentials for a mixture of alkanes in the oil industry, the Lagrangian manifold. It can be noted that three critical phenomena have been discovered for the phase transition.

We also recommend that you read the article:
[Beattie J.A., Bridgeman O.C. An Equation of State for Gaseous Mixtures. 1. Application to Mixtures of Methane and Nitrogen // J. Am. Chem. Soc. 1929. V. 51. P. 19–30.](https://doi.org/10.1021/ja01376a003)

`MSLVMix` The model describes a substance state in three phases. Thermodynamics states are points on Legendrian or Lagrangian manifolds in the corresponding contact or symplectic spaces in terms of differential geometry. The conditions of applicable states and the first order phase transition are given for the Modified Solid-Liquid-Vapour equation of state. The Lagrangian manifold, singularity curve and the phase transition curves are plotted for methane. The description of the method is in the following article: 
[Maksim Kostiuchek, Alexey Batov, Ivan Galyaev, Anton Salnikov. Some Features of the Modified Solid-Liquid-Vapor Equation of State.](http://dx.doi.org/10.2139/ssrn.4613860)


...

## Example scripts
`Run_Beattie-Bridgeman` Beattie-Bridgeman example.
`MSLVMix_expl` 

## How to install and run
If you would like to modify the source code, download the feslib repository and install using pip (or pip3 depending on your installation).
```bash
    git clone https://github.com/LychaginTeam/feslib.git
    cd feslib/
    pip install .
```
Alternatively, you can install feslib via pip
```bash
   pip install feslib
```

## To run the example scripts
```bash
    cd examples
    python MSLVMix_expl.py
```

## Authors
[Maksim Kostiuchek](https://www.ipu.ru/node/47150) (max31@list.ru),
[Alexey Batov](https://www.ipu.ru/node/82) (batov@ipu.ru),
[Anton Salnikov](https://www.ipu.ru/staff/salnikov) (salnikov@ipu.ru),
[Ivan Galyaev](https://www.ipu.ru/node/49970) (ivan.galyaev@yandex.ru),
[Valentin Lychagin](https://www.ipu.ru/node/457) (valentin.lychagin@uit.no)

## Cite
You can cite the latest release of the package as:
feslib: 0.1.0 (Version 0.1.0). Zenodo. http://doi.org/...

## Acknowledgments
The development of this library was supported by Russian Science Foundation grant number 21-71-20034.
