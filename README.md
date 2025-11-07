# feslib

Thermodynamics, phase transitions, applications for different fluids, equations of state.

## Description

**feslib** provides several functions and example scripts for investigation of phase transitions.
Several different equations of state are implemented in the library.
The library is based on the results of articles by Valentin Lychagin, his students and colleagues.

Useful bibliography:

 [Valentin Lychagin Thermodynamics as a theory of measurement](https://doi.org/10.1016/j.geomphys.2021.104430)


#### Contribute
This code is still under development and benchmarking. If you find any bugs or errors in the code, please report them in GitHub.

## Methods
`Beattie-Bridgeman` Phase transitions using Beattie-Bridgeman EOS. The Beattie–Bridgeman model was chosen to describe phase transitions and their features for the equation of state of a real gas. The description of the method is in the following article: 
[I. A. Galyaev, M. I. Kostiuchek, A. V. Batov, and A. M. Salnikov. Critical Phenomena of Massieu–Plank Potential for Gas Mixtures Described by the Beattie–Bridgeman Equations of State // Lobachevskii Journal of Mathematics, 2023, Vol. 44, No. 9, pp. 3919–3926](https://doi.org/10.1134/S1995080223090093)

The paper considers thermodynamics as a measurement of extensive variables, such as energy, volume and mass. In this sense, thermodynamic states are Legendrian or Lagrangian surfaces in the corresponding contact or symplectic space. The Beattie-Bridgeman model was chosen to describe phase transitions and their features for the equation of state of a real gas. This model describes the state of a substance in two phases: liquid and vapor. Real gas can be either single gas or a mixture. The program provides calculated data for many gas mixtures: methane, ethane, propane, butane, pentane, hydrogen, nitrogen, carbon dioxide, ammonia. Using the program, you can calculate formulas for model constants for any mixture of gases. You can obtain graphs of: the caloric equation of state and various phase transition potentials for a mixture of alkanes in the oil industry, the Lagrangian manifold. It can be noted that three critical phenomena have been discovered for the phase transition.

We also recommend that you read the article:
[Beattie J.A., Bridgeman O.C. An Equation of State for Gaseous Mixtures. 1. Application to Mixtures of Methane and Nitrogen // J. Am. Chem. Soc. 1929. V. 51. P. 19–30.](https://doi.org/10.1021/ja01376a003)

`MSLVMix` The model describes a substance state in three phases. Thermodynamic states are points on Legendrian or Lagrangian manifolds in the corresponding contact or symplectic spaces in terms of differential geometry. The conditions of applicable states and the first order phase transition are given for the Modified Solid-Liquid-Vapour equation of state. The Lagrangian manifold, singularity curve and the phase transition curves are plotted for methane. The description of the method is in the following article: 
[Batov, A. V., Galyaev, I. A., Kostiuchek, M. I. & Salnikov, A. M. *Some Features of the Modified Solid–Liquid–Vapor Equation of State*. Lobachevskii J Math 45, 1905–1916 (2024)].(https://doi.org/10.1134/S1995080224602078)

In addition to these models, `feslib` also provides tools to work with virial equations of state and virial coefficients for real gases. The `appr_example.ipynb` notebook and the data in the `data/` directory demonstrate how to fit the virial coefficients \(B_1(T)\) and \(B_2(T)\) for methane using data from the `data/` directory.
[Batov, A., Kostiuchek, M., Salnikov, A. & Galyaev, I. *Using the Virial Equation of State to Approximate Methane Data*. in 2024 17th International Conference on Management of Large-Scale System Development (MLSD) 1–4 (IEEE, Moscow, Russian Federation, 2024).](doi:10.1109/MLSD61779.2024.10739643.)


## Example scripts
`Run_Beattie-Bridgeman` -- example for Beattie-Bridgeman equation model.

`MSLVMix_expl` -- example for MSLV equation model.

`VdWMix_example` -- example of calculating mixtures using the Van der Waals equation.

`appr_example.ipynb` -- example of these coefficients using polynomials 

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
    python VdWMix_example.py
```

## Authors
[Maksim Kostiuchek](https://www.ipu.ru/node/47150) (max31@list.ru),
[Alexey Batov](https://www.ipu.ru/node/82) (batov@ipu.ru),
[Anton Salnikov](https://www.ipu.ru/staff/salnikov) (salnikov@ipu.ru),
[Ivan Galyaev](https://www.ipu.ru/node/49970) (ivan.galyaev@yandex.ru),
[Valentin Lychagin](https://www.ipu.ru/node/457) (valentin.lychagin@uit.no)

## How to cite

If you use this library in your research, please cite the following works:

- Lychagin, V. *Thermodynamics as a theory of measurement*. Journal of Geometry and Physics 172, 104430 (2022). https://doi.org/10.1016/j.geomphys.2021.104430
- Galyaev, I.A., Kostiuchek, M.I., Batov, A.V. et al. *Critical Phenomena of Massieu–Plank Potential for Gas Mixtures Described by the Beattie–Bridgeman Equations of State*. Lobachevskii J Math 44, 3919–3926 (2023). https://doi.org/10.1134/S1995080223090093.
- Batov, A. V., Galyaev, I. A., Kostiuchek, M. I. & Salnikov, A. M. *Some Features of the Modified Solid–Liquid–Vapor Equation of State*. Lobachevskii J Math 45, 1905–1916 (2024). https://doi.org/10.1134/S1995080224602078.
- Batov, A., Kostiuchek, M., Salnikov, A. & Galyaev, I. *Using the Virial Equation of State to Approximate Methane Data*. in 2024 17th International Conference on Management of Large-Scale System Development (MLSD) 1–4 (IEEE, Moscow, Russian Federation, 2024). doi:10.1109/MLSD61779.2024.10739643.
