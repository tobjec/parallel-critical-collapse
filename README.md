# Critical Gravitational Collapse Simulator

This project provides a modern `C++` implementation of the critical gravitational collapse of a massless scalar field in spherical symmetry. It focuses on constructing discretely self-similar solutions in rational spacetime dimensions between 3 and 5. 

The framework builds upon spectral methods in logarithmic time and finite-difference integration in space, combined with a Newton–shooting method to solve the resulting non-linear boundary value problem. It supports serial as well as parallel execution using OpenMP, MPI, or a hybrid of both, enabling efficient large-scale simulations on HPC systems.

---

## Building

The code supports four different build modes, configured via CMake options:

- **Serial (`cc_serial`)** – pure sequential run (always available).
- **OpenMP (`cc_openmp`)** – shared-memory parallelism on multicore CPUs.
- **MPI (`cc_mpi`)** – distributed-memory parallelism across nodes.
- **Hybrid (`cc_hybrid`)** – combined MPI + OpenMP execution.

### Dependencies

The project relies on the following libraries:

- **[CMake](https://www.cmake.org/) (≥ 3.20)** – build system
- **[FFTW3](http://www.fftw.org/)** – spectral Fourier transforms
- **[LAPACK](https://www.netlib.org/lapack/)** – linear algebra routines
- **[nlohmann/json](https://github.com/nlohmann/json)** – JSON for Modern C++
- **[OpenMP](https://www.openmp.org/)** (optional) – shared-memory parallelism
- **[MPI](https://www.mpi-forum.org/)** (optional) – distributed-memory parallelism

### Installation on Ubuntu

Update your package list and install the required development packages:

```bash
sudo apt update
sudo apt install -y build-essential cmake pkg-config \
    libfftw3-dev liblapacke-dev \
    libopenmpi-dev openmpi-bin \
    libomp-dev
```

For the JSON library, you can either install via package manager:

```bash
sudo apt install nlohmann-json3-dev
```

or fetch it with CMake’s `FetchContent` if you prefer a header-only inclusion.

Build example:

```bash
mkdir -p build && cd build
cmake -DENABLE_OPENMP=ON -DENABLE_MPI=ON -DENABLE_HYBRID=ON -DENABLE_SERIAL=ON ..
make -j
```

Executables are placed in the build directory, e.g. `cc_serial`, `cc_openmp`, `cc_mpi`, `cc_hybrid`.

---

## Command Line Parameters

Each executable accepts the following arguments:

- `-s, --single-run`  
  Run a single simulation (default).

- `-m, --multiple-run`  
  Run multiple simulations from a JSON input dictionary.

- `--ignore-converged`  
  In multiple-run mode, skip already converged simulations.

- `-i, --input-path <path>`  
  Path to a simulation input JSON file.  
  Default: `data/simulation_4D_512.json`

- `-r, --reversed-order`  
  Reverse the execution order of simulation dimensions.

- `-b, --benchmark`  
  Enable benchmark mode (repeated runs of the same simulation).

- `--benchmark-repetitions <N>`  
  Set number of repetitions in benchmark mode. Default: `3`.

---

## Examples

### Run a single simulation

```bash
./cc_serial -i data/simulation_4D_512.json
```

### Run multiple simulations from a multidimensional input file
```bash
export OMP_NUM_THREADS=8
./cc_openmp -m -i data/simulation_data.json
```

### Run multiple simulations and also recalculate already converged ones
```bash
mpirun -np 8 ./cc_mpi -m --ignore-converged -i data/simulation_data.json
```

### Run multiple simulations in reversed order (low to high dimension)
```bash
export OMP_NUM_THREADS=2
mpirun -np 4 ./cc_hybrid -m -r -i data/simulation_data.json
```

### Benchmark a single simulation with 5 repetitions
```bash 
export OMP_NUM_THREADS=2
mpirun -np 4 ./cc_hybrid -i data/simulation_4D_512.json -b --benchmark-repetitions 5
```

---

## Scripts

### `plot_simulation_results.py`

Create publication-ready figures from simulation JSON output.

**Synopsis**
```bash
python3 scripts/plot_simulation_results.py -i <FILES...> -o <OUT> -k <KIND> [--spec <SPEC>] [-e <EXP>] [-d <DIM>] [-s]
```

#### Options

- `-i, --input_files`  
  Input JSON files (one or more).  
  *Default:* predefined convergence test files.

- `-o, --output_name`  
  Output path and base name for plots, meaningful name will be added automatically.  
  *Default:* `data/cc_plot.pdf`

- `-e, --experimental_data`  
  Optional path to experimental or reference data for comparison.

- `-k, --kind`  
  Type of plot to generate.  
  Choices: `convergence`, `fields`, `initial_data`, `echoing_period`, `benchmark`, `mismatch_layer_finder`, `theoretical_speedup`, `efficiency`.  
  *Default:* `convergence`

- `--spec`  
  Plot specification/refinement.  
  Choices: `3R`, `differences`, `vminu`, `rel_su`.

- `-d, --dim`  
  Dimension to be postprocessed (if applicable).

- `-s, --single_plots`  
  Produce individual plots instead of grid plots.

#### Examples

- Generate a convergence plot (requires 5 input files):
```bash
python3 scripts/plot_simulation_results.py \
  -k convergence \
  -i data/simulation_convergence_base.json \
     data/simulation_convergence_xcut02.json \
     data/simulation_convergence_xcut04.json \
     data/simulation_convergence_xcut12.json \
     data/simulation_convergence_xcut14.json \
  -o data/cc_plot.pdf
```
- Plot initial data for a single dimension with reference comparison:
```python
python3 scripts/plot_simulation_results.py \
  -k initial_data -d 3.750 \
  -i data/simulation_data.json \
  -e data/fortran_reference.json \
  -o data/cc_plot.pdf
```

### `create_simulation_data.py`

Generate or modify simulation input JSON files (multi-D expansion, Ntau doubling, merging).

**Synopsis**
```bash
python3 scripts/create_simulation_data.py -i <IN...> -o <OUT...> -k <KIND> [-d <DIM>] [-r]
```

#### Options

- `-i, --input`  
  Input JSON file(s).  
  *Default:* `../data/simulation_config.json`

- `-o, --output`  
  Output JSON file(s).  
  *Default:* `../data/simulation_data.json`

- `-k, --kind`  
  Operation: `create_multidim`, `double_nt`, `merge`.  
  *Default:* `create_multidim`

- `-d, --dim`  
  Dimension to modify (used with `double_nt`).

- `-r, --reversed`  
  Reverse ordering or update direction.

#### Examples

- Expand template to multi-D simulation data:
```python3
python3 scripts/create_simulation_data.py \
  -k create_multidim \
  -i data/simulation_config.json \
  -o data/simulation_data.json
```
- Double Ntau for dimension 3.750:
```python3
python3 scripts/create_simulation_data.py \
  -k double_nt -d 3.750 \
  -i data/simulation_data.json \
  -o data/simulation_data_doubled.json
```

--- 

## Tests

Unit tests are included under `test/` and can be run via:

```bash
ctest
```

These validate FFT operations, LAPACK integration, and initial condition generation.

---

## License

This project is licensed under the MIT License – see the [LICENSE](LICENSE) file for details.

