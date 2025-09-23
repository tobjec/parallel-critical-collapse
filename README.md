# Critical Gravitational Collapse Simulator

This project provides a modern C++ implementation of the critical gravitational collapse of a massless scalar field in spherical symmetry. It focuses on constructing discretely self-similar solutions in rational spacetime dimensions between 3 and 5. The framework builds upon spectral methods in logarithmic time and finite-difference integration in space, combined with a Newton–shooting method to solve the resulting non-linear boundary value problem. It supports serial as well as parallel execution using OpenMP, MPI, or a hybrid of both, enabling efficient large-scale simulations on HPC systems.

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


