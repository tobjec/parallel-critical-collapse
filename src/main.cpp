//==============================================================================
// main.cpp
// Entry point for running critical-collapse simulations.
// Modes:
//   - Single run (default) over one JSON config.
//   - Multiple run (sweep over target dimensions inside a suite JSON).
//   - Benchmark mode (repeat runs and write timing/results summary).
//
// Parallel backends (compile-time):
//   USE_MPI     : pure MPI
//   USE_OPENMP  : shared-memory threading
//   USE_HYBRID  : MPI + OpenMP (MPI_Init_thread with FUNNELED)
//==============================================================================

#include "common.hpp"
#include "SimulationConfig.hpp"
#include "NewtonSolver.hpp"

int main(int argc, char* argv[])
{
    //--------------------------------------------------------------------------
    // CLI flags (defaults)
    //   -s/--single-run        : run a single simulation (default)
    //   -m/--multiple-run      : run a sweep over dimensions from a suite file
    //   --ignore-converged     : treat configs as unconverged (force re-solve)
    //   -i/--input-path <path> : JSON input (single or suite)
    //   -r/--reversed-order    : traverse suite dimensions in reverse
    //   -b/--benchmark         : enable benchmark mode
    //   --benchmark-repetitions <n> : repetitions for benchmark (default 3)
    //--------------------------------------------------------------------------
    bool singleRun = true;
    bool ignoreConverged = false;
    bool reversed = false;
    bool benchmark = false;
    int  benchmark_repetitions = 3;
    std::string inputPath{"data/simulation_4D_512.json"};

    // Parse CLI args (very lightweight; no error if unknown flag)
    if (argc > 1)
    {
        for (int i = 1; i < argc; ++i)
        {
            std::string arg = argv[i];

            if (arg == "--single-run" || arg == "-s")
            {
                singleRun = true;
            }
            else if (arg == "--multiple-run" || arg == "-m")
            {
                singleRun = false;
            }
            else if (arg == "--ignore-converged")
            {
                ignoreConverged = true;
            }
            else if (arg == "--input-path" || arg == "-i")
            {
                // Expect a path argument right after the flag
                if (i+1 < argc) { inputPath = std::string(argv[i+1]); }

                // Fail fast if file does not exist
                if (!std::filesystem::exists(inputPath))
                {
                    throw std::invalid_argument("Invalid simulation input path!");
                }
            }
            else if (arg == "--reversed-order" || arg == "-r")
            {
                reversed = true;
            }
            else if (arg == "--benchmark" || arg == "-b")
            {
                benchmark = true;
            }
            else if (arg == "--benchmark-repetitions")
            {
                benchmark = true;
                if (i+1 < argc) { benchmark_repetitions = std::stoi(argv[i+1]); }
            }
        }
    }

    //--------------------------------------------------------------------------
    // Derive data directory from input file (absolute path, parent folder).
    // Used to store outputs (e.g., benchmark_*.json).
    //--------------------------------------------------------------------------
    std::filesystem::path dataPath(inputPath);
    dataPath = std::filesystem::absolute(dataPath);
    dataPath = dataPath.parent_path();

    //--------------------------------------------------------------------------
    // Initialize parallel runtime depending on backend.
    //  - USE_MPI   : MPI_Init
    //  - USE_HYBRID: MPI_Init_thread with FUNNELED support (abort if missing)
    //  - otherwise : serial
    //--------------------------------------------------------------------------
    #if defined(USE_MPI)
    MPI_Init(&argc, &argv);
    #elif defined(USE_HYBRID)
    int required = MPI_THREAD_FUNNELED;
    int provided = -1;
    MPI_Init_thread(&argc, &argv, required, &provided);
    if (provided < required)
    {
        int rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        if (rank == 0) std::cout << "Not enough thread support!" << std::endl;
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
    #endif

    // Rank/size setup; serial uses rank=0 for uniform logging
    #if defined(USE_MPI) || defined(USE_HYBRID)
    int rank, size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    #else
    int rank = 0;
    #endif

    try
    {
        //======================================================================
        // BENCHMARK MODE
        // Repeat Newton solves to collect timing/statistics into a JSON file.
        // Output filename encodes backend and resources (cores/threads).
        //======================================================================
        if (benchmark)
        {
            json benchmark_results;
            SimulationConfig config = SimulationConfig::loadFromJson(inputPath);
            if (ignoreConverged) config.Converged = false;

            // Header metadata
            benchmark_results["Dim"] = config.Dim;
            benchmark_results["Ntau"] = config.Ntau;
            benchmark_results["XLeft"] = config.XLeft;
            benchmark_results["XMid"] = config.XMid;
            benchmark_results["XRight"] = config.XRight;
            benchmark_results["PrecisionNewton"] = config.PrecisionNewton;
            benchmark_results["NLeft"] = config.NLeft;
            benchmark_results["NRight"] = config.NRight;
            benchmark_results["Repetitions"] = benchmark_repetitions;

            // Choose output filename by backend
            #if defined(USE_MPI)
            benchmark_results["Kind"] = "MPI";
            benchmark_results["Cores"] = size;
            auto benchmarkOutputPath = dataPath / ("benchmark_D=" + std::to_string(config.Dim)
                                       + "_MPI_" + std::to_string(size) + ".json");
            #elif defined(USE_HYBRID)
            benchmark_results["Kind"] = "Hybrid";
            benchmark_results["Cores"] = size;
            benchmark_results["Threads"] = omp_get_max_threads();
            auto benchmarkOutputPath = dataPath / ("benchmark_D=" + std::to_string(config.Dim)
                                       + "_Hybrid_" + std::to_string(size) + "_"
                                       + std::to_string(omp_get_max_threads()) + ".json");
            #elif defined(USE_OPENMP)
            benchmark_results["Kind"] = "OpenMP";
            benchmark_results["Threads"] = omp_get_max_threads();
            auto benchmarkOutputPath = dataPath / ("benchmark_D=" + std::to_string(config.Dim)
                                       + "_OpenMP_" + std::to_string(omp_get_max_threads()) + ".json");
            #else
            benchmark_results["Kind"] = "Serial";
            benchmark_results["Cores"] = 1;
            auto benchmarkOutputPath = dataPath / ("benchmark_D=" + std::to_string(config.Dim)
                                       + "_Serial.json");
            #endif

            // Visibility of progress
            #if defined(USE_MPI) || defined(USE_HYBRID)
            if (rank==0) { std::cout << "Starting benchmark run for D=" << config.Dim << ".\n\n"; }
            #else
            std::cout << "Starting benchmark run for D=" << config.Dim << ".\n\n";
            #endif

            // Repetitions: collect per-run data under stringified index keys
            for (int i=0; i<benchmark_repetitions; ++i)
            {
                #if defined(USE_MPI) || defined(USE_HYBRID)
                if (rank==0)
                    std::cout << "Repetition " << i+1 << "/" << benchmark_repetitions << "\n\n";
                #else
                std::cout << "Repetition " << i+1 << "/" << benchmark_repetitions << "\n\n";
                #endif

                NewtonSolver solver(config, dataPath, benchmark);
                solver.run(&benchmark_results[std::to_string(i)]);
            }

            // Root rank writes the benchmark summary to disk
            #if defined(USE_MPI) || defined(USE_HYBRID)
            if (rank == 0)
            {
                std::cout << "Benchmark result stored in file: " << benchmarkOutputPath << "\n\n";
                OutputWriter::writeJsonToFile(benchmarkOutputPath.c_str(), benchmark_results);
            }
            #else
            std::cout << "Benchmark result stored in file: " << benchmarkOutputPath << "\n\n";
            OutputWriter::writeJsonToFile(benchmarkOutputPath.c_str(), benchmark_results);
            #endif
        }
        //======================================================================
        // SINGLE RUN
        // Load one SimulationConfig from JSON, run NewtonSolver once, and write
        // results back into the same JSON file.
        //======================================================================
        else if (singleRun)
        {
            json result;
            SimulationConfig config = SimulationConfig::loadFromJson(inputPath);
            if (ignoreConverged) config.Converged = false;

            #if defined(USE_MPI) || defined(USE_HYBRID)
            if (rank==0) { std::cout << "Starting single run for D=" << config.Dim << ".\n\n"; }
            #else
            std::cout << "Starting single run for D=" << config.Dim << ".\n\n";
            #endif

            NewtonSolver solver(config, dataPath);
            result = solver.run();

            // Persist result (root-only in MPI/Hybrid)
            #if defined(USE_MPI) || defined(USE_HYBRID)
            if (rank == 0)
            {
                std::cout << "Result stored in file: " << inputPath << "\n\n";
                OutputWriter::writeJsonToFile(inputPath, result);
            }
            #else
            std::cout << "Result stored in file: " << inputPath << "\n\n";
            OutputWriter::writeJsonToFile(inputPath, result);
            #endif
        }
        //======================================================================
        // MULTIPLE RUN (SUITE)
        // Iterate over a set of target dimensions specified in a suite JSON.
        // For each target D:
        //   - If initial data (fc, Up, psic) is missing, try borrowing from the
        //     previous converged run(s). If ≥3 previous converged dimensions
        //     exist, perform quadratic extrapolation in D for {fc,Up,psic,Δ}.
        //   - Run Newton solver and accumulate results into one JSON dictionary.
        //======================================================================
        else
        {
            SimulationSuite configSuite(inputPath, reversed, ignoreConverged);

            #if defined(USE_MPI) || defined(USE_HYBRID)
            if (rank==0)
                std::cout << "Starting multi run for " << configSuite.simulationDims.size()
                          << " distinct dimensions.\n\n";
            #else
            std::cout << "Starting multi run for " << configSuite.simulationDims.size()
                      << " distinct dimensions.\n\n";
            #endif

            for (size_t i=0; i<configSuite.simulationDims.size(); ++i)
            {
                std::string simDim = configSuite.simulationDims[i];
                SimulationConfig config = configSuite.generateSimulation(simDim);

                #if defined(USE_MPI) || defined(USE_HYBRID)
                if (rank==0)
                {
                    std::cout << "Simulation " << i+1 << "/" << configSuite.simulationDims.size() << ":\n";
                    std::cout << "Starting simulation for D=" << simDim << ".\n\n";
                }
                #else
                std::cout << "Simulation " << i+1 << "/" << configSuite.simulationDims.size() << ":\n";
                std::cout << "Starting simulation for D=" << simDim << ".\n\n";
                #endif

                //--- Initial data policy --------------------------------------
                // i=0: must have initial data in the suite, or fail.
                if ((config.fc.empty() || config.Up.empty() || config.psic.empty()) && i==0)
                {
                    throw std::runtime_error("No converged solution as initial conditions available for " + simDim + "!");
                }
                // i=1 or i=2: try to copy directly from previous dimension if converged
                else if ((config.fc.empty() || config.Up.empty() || config.psic.empty()) && i>0 && i<3)
                {
                    std::string prevDim = configSuite.simulationDims[i-1];
                    SimulationConfig config_prev = configSuite.generateSimulation(prevDim);

                    if (config_prev.Converged)
                    {
                        // Borrow Δ, fc, Up, psic from previous converged solution
                        config.Delta = config_prev.Delta;
                        config.fc    = config_prev.fc;
                        config.Up    = config_prev.Up;
                        config.psic  = config_prev.psic;
                    }
                    else
                    {
                        throw std::runtime_error("Previous dimension "+ prevDim +" is not converged, hence no initial data for next simulation!");
                    }
                }
                // i>=3: do quadratic extrapolation in D using the last 3 converged runs
                else if (config.fc.empty() || config.Up.empty() || config.psic.empty())
                {
                    std::string prevDim1 = configSuite.simulationDims[i-1];
                    std::string prevDim2 = configSuite.simulationDims[i-2];
                    std::string prevDim3 = configSuite.simulationDims[i-3];

                    SimulationConfig config_prev1 = configSuite.generateSimulation(prevDim1);
                    SimulationConfig config_prev2 = configSuite.generateSimulation(prevDim2);
                    SimulationConfig config_prev3 = configSuite.generateSimulation(prevDim3);

                    // Need all three to be converged to build quadratic model
                    if (!config_prev1.Converged || !config_prev2.Converged || !config_prev3.Converged)
                    {
                        throw std::runtime_error("Previous dimensions not converged, hence no initial data for next simulation!");
                    }

                    // Collect previous field arrays
                    vec_real fc1 = config_prev1.fc,  fc2 = config_prev2.fc,  fc3 = config_prev3.fc;
                    vec_real Up1 = config_prev1.Up,  Up2 = config_prev2.Up,  Up3 = config_prev3.Up;
                    vec_real ps1 = config_prev1.psic,ps2 = config_prev2.psic,ps3 = config_prev3.psic;

                    vec_real Deltas = {config_prev1.Delta, config_prev2.Delta, config_prev3.Delta};

                    // Targets (allocate zeros; sizes match previous runs)
                    vec_real fc(fc1.size(),0), Up(Up1.size(),0), psic(ps1.size(),0);
                    real_t  Delta{};

                    // Quadratic fit in dimension D for each τ sample independently
                    vec_real dims = {std::stod(prevDim1), std::stod(prevDim2), std::stod(prevDim3)};
                    real_t  curr_dim = std::stod(simDim);

                    for (size_t j=0; j<fc1.size(); ++j)
                    {
                        vec_real y_fc   = {fc1[j],  fc2[j],  fc3[j]};
                        vec_real y_Up   = {Up1[j],  Up2[j],  Up3[j]};
                        vec_real y_psic = {ps1[j],  ps2[j],  ps3[j]};

                        // Solve [a,b,c] = argmin ||[D^2 D 1]*[a b c]^T - y||₂ using LAPACK
                        vec_real coeff_fc   = fit_quadratic_least_squares(dims, y_fc);
                        vec_real coeff_Up   = fit_quadratic_least_squares(dims, y_Up);
                        vec_real coeff_psic = fit_quadratic_least_squares(dims, y_psic);

                        fc[j]   = coeff_fc[0]*curr_dim*curr_dim   + coeff_fc[1]*curr_dim   + coeff_fc[2];
                        Up[j]   = coeff_Up[0]*curr_dim*curr_dim   + coeff_Up[1]*curr_dim   + coeff_Up[2];
                        psic[j] = coeff_psic[0]*curr_dim*curr_dim + coeff_psic[1]*curr_dim + coeff_psic[2];
                    }

                    // Extrapolate Δ the same way
                    vec_real coeff_Delta = fit_quadratic_least_squares(dims, Deltas);
                    Delta = coeff_Delta[0]*curr_dim*curr_dim + coeff_Delta[1]*curr_dim + coeff_Delta[2];

                    // Store extrapolated initial data
                    config.Delta = Delta;
                    config.fc    = fc;
                    config.Up    = Up;
                    config.psic  = psic;
                }

                // Optionally override converged flag (force new solve)
                if (ignoreConverged) config.Converged = false;

                // Run solver and accumulate results under the dimension key
                NewtonSolver solver(config, dataPath);
                configSuite.multiInputDict[simDim] = solver.run();

                // Persist growing multi-dictionary (root-only if parallel)
                #if defined(USE_MPI) || defined(USE_HYBRID)
                if (rank == 0)
                {
                    std::cout << "Result stored in file: " << inputPath << "\n\n";
                    OutputWriter::writeJsonToFile(inputPath, configSuite.multiInputDict);
                }
                MPI_Barrier(MPI_COMM_WORLD);
                #else
                std::cout << "Result stored in file: " << inputPath << "\n\n";
                OutputWriter::writeJsonToFile(inputPath, configSuite.multiInputDict);
                #endif
            }
        }

        //--------------------------------------------------------------------------
        // Normal termination & finalize MPI if needed.
        //--------------------------------------------------------------------------
        #if defined(USE_MPI) || defined(USE_HYBRID)
        if (rank==0) { std::cout << "Simulation finished successfully.\n\n"; }
        MPI_Finalize();
        #else
        std::cout << "Simulation finished successfully.\n\n";
        #endif

        return 0;
    }
    catch (const std::exception& e)
    {
        // Root-cause visibility; in MPI paths, abort all ranks.
        std::cerr << "Rank " << rank << " caught exception: " << e.what() << std::endl;

        #if defined(USE_MPI) || defined(USE_HYBRID)
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        #endif

        return 1;
    }
}
