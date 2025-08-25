# Jipole

A Julia-based radiative transfer code for curved spacetimes with automatic differentiation capabilities.

## Overview

Jipole is an ipole-based Julia implementation designed to perform radiative transfer calculations in curved spacetimes, with a particular focus on black hole imaging. The code leverages Julia's automatic differentiation (autodiff) to compute derivatives of input parameters, enabling gradient based optimization methods. For this project, we used [ForwardDiff.jl](https://github.com/JuliaDiff/ForwardDiff.jl) ([Revels et al. 2016](https://arxiv.org/abs/1607.07892))


## Current Development Status

The current version of Jipole focuses on implementing and validating the test problems described in **Section 3.2** of [Gold et al. 2020](https://iopscience.iop.org/article/10.3847/1538-4357/ab96c6). This development phase involves porting the well-established C-based[ipole](https://github.com/moscibrodzka/ipole) ([Moscibrodzka & Gammie 2017](https://arxiv.org/abs/1712.03057)) code to Julia with extended functionality.


## Installation and Setup

### First-Time Setup

If this is your first time using Jipole, you'll need to set up the Julia environment and install the required dependencies:

```julia
using Pkg
Pkg.instantiate()
```

This command will install all the packages specified in the `Project.toml` and `Manifest.toml` files.

### Jupyter Kernel Installation

To use Jipole with Jupyter notebooks, install the project-specific kernel:

```julia
using IJulia
IJulia.installkernel("Jipole", "--project=" * Base.current_project())
```

This creates a dedicated Jupyter kernel that automatically loads the Jipole project environment.

## Configuration

### Model Parameters

The analytical model parameters are configured in `./src/models/analytic.jl`. This file contains the fundamental physical parameters that define your radiative transfer model:

```julia
const A = 1.e6          # Absorption parameter
const α_analytic = -0.0 # Emissivity's exponential dependence on frequency
const height = (100. / 3.0)  # Disk height parameter
const l0 = 1.0          # whether to consider gas with angular momentum
```

#### Parameter Descriptions

- **A**: Absorption parameter that controls the overall absorption coefficient in the medium
- **α_analytic**: Controls the emissivity's exponential dependence on frequency (spectral index)
- **height**: Disk height parameter defining the vertical structure of the accretion disk
- **l0**: Boolean-like parameter determining whether to consider gas with angular momentum effects

The black hole spin parameter can be adjusted directly within the Jupyter notebooks, allowing for interactive exploration of different spacetime geometries without modifying source code.

## Running Jipole

### Starting the Environment

1. **Navigate to the project directory**:
   ```bash
   cd /path/to/jipole
   ```

2. **Start Julia with multithreading capabilities**:
   ```bash
   JULIA_NUM_THREADS=xx julia --project="."
   ```
   
   Replace `xx` with the number of CPU cores you want to utilize.

3. **Launch JupyterLab**:
   ```julia
   using IJulia
   IJulia.jupyterlab(dir=pwd())
   ```

### Using Jupyter Notebooks

1. Open your web browser and navigate to the JupyterLab interface (typically `http://localhost:8888`)
2. When creating or opening a notebook, ensure you select the **"Jipole"** kernel from the kernel menu
3. The Jipole kernel ensures that all Jipole dependencies are automatically loaded

## References

- Gold, R. et al. 2020, ApJ, 897, 148: "Verification of Radiative Transfer Schemes for the EHT"
- Moscibrodzka, M. & Gammie, C. F. 2017, "ipole - semianalytic scheme for relativistic polarized radiative transport", arXiv:1712.03057
- Revels, J., Lubin, M., and Papamarkou, T. 2016, "Forward-Mode Automatic Differentiation in Julia", arXiv:1607.07892


## Contact

Reach me at pedronaethemotta@usp.br