# Critical Collapse Simulation

This project implements a modern C++ rewrite of a classical Fortran77 code to simulate the critical gravitational collapse of a scalar field in spherical symmetry using spectral methods and implicit Runge-Kutta time integration.

---

## 🚀 Features

- Spectral Fourier-based solver with anti-aliasing  
- High-order Taylor expansions for initial data  
- Fully implicit and adaptive Runge-Kutta integrators (IRK2, RKF45)  
- Full Newton-Raphson solver with automatic Jacobian assembly  
- FFTW3 and LAPACKE backend support  
- Configurable via clean JSON input  
- All results stored in a unified `results.json` file  

---

## 📦 Build Instructions

### Requirements

- C++17 compiler (e.g. `g++`, `clang++`)  
- `cmake ≥ 3.16`  
- `FFTW3`, `LAPACKE`, and `OpenMP` installed (e.g., via your package manager)

### Building

```bash
mkdir -p build && cd build
cmake ..
make -j
```

The executable will be placed in `./build/bin/critical_collapse`.

---

## ⚙️ Running the Simulation

Prepare your simulation parameters in:

```
data/simulation.json
```

### Example

```json
{
  "Ny": 256,
  "Dim": 4.0,
  "XLeft": 0.001,
  "XMid": 0.1,
  "XRight": 0.999,
  "EpsNewton": 1e-8,
  "PrecisionNewton": 1e-14,
  "Verbose": true,
  "SlowError": 1e-3,
  "OutEvery": 0,
  "UseLogGrid": false,
  "NLeft": 3000,
  "NRight": 3000,
  "Tolerance": 1e-15,
  "TimeStep": 2,
  "PrecisionIRK": 1e-15
}
```

### Run

```bash
./bin/critical_collapse
```

Results will be appended to:

```
data/results.json
```

with keys like `"4.0"`, `"3.9"` identifying the dimension used.

---

## 🧪 Tests (WIP)

A `test/` folder is provided for future automated validation using Catch2 or GoogleTest.

---

## 📁 Project Structure

```
.
├── include/        # Headers
├── src/            # Source files
├── data/           # Input/output files
├── build/          # Build directory
├── test/           # Unit tests (optional)
├── CMakeLists.txt
├── simulation.json # Sample input
└── results.json    # Collected simulation output
```

---

## 📜 License

This project is released under the MIT License.

---

## ✏️ Contributors

- Lead Developer: Tobias Jechtl
- Project Supervisers: Florian Ecker, Daniel Grumiller
- Based on theoretical work of Choptuik, Garfinkle, and collaborators.

---

## 🧠 References

- M. W. Choptuik, “Universality and Scaling in Gravitational Collapse of a Massless Scalar Field,” *Phys. Rev. Lett.* **70**, 9 (1993).  
- C. Gundlach, “Critical phenomena in gravitational collapse,” *Living Reviews in Relativity* **2**, 4 (1999).