#include "common.hpp"
#include "SimulationConfig.hpp"
#include "NewtonSolver.hpp"

int main(int argc, char* argv[])
{
    bool singleRun = true;
    bool ignoreConverged = false;
    bool reversed = false;
    std::string inputPath{};
    std::string firstDim{"4.000"};

    if (argc<2)
    {
        inputPath = "../data/simulation_4D.json";
    }
    else
    {

        for (int i=1; i<argc; ++i)
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
                if (i+1 < argc)
                {
                    inputPath = std::string(argv[i+1]);
                }

                if (!std::filesystem::exists(inputPath))
                {
                    throw std::invalid_argument("Invalid simulation input path!");
                }
            }
            else if (arg == "--first-dim" || arg == "-d")
            {
                if (i+1 < argc)
                {
                    firstDim = std::string(argv[i+1]);
                }
            }
            else if (arg == "--reversed-order" || arg == "-r")
            {
                reversed = true;
            }
        }

    }

    std::filesystem::path dataPath(inputPath);
    dataPath = std::filesystem::absolute(dataPath);
    dataPath = dataPath.parent_path();

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
        if (rank==0) std::cout << "Not enough thread support!" << std::endl;
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    } 
    #endif

    #if defined(USE_MPI) || defined(USE_HYBRID)
    int rank, size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    #else
    int rank = 0;
    #endif

    try
    {
        if (singleRun)
        {
            json result;
            // Load configuration
            SimulationConfig config = SimulationConfig::loadFromJson(inputPath);

            if (ignoreConverged) config.Converged = false;

            #if defined(USE_MPI) || defined(USE_HYBRID)
            if (rank==0)
            {
                std::cout << "Starting single run for D=" << config.Dim << "." << std::endl << std::endl;
            }
            #else
            std::cout << "Starting single run for D=" << config.Dim << "." << std::endl << std::endl;
            #endif

            // Instantiate Newton solver with essential parameters
            NewtonSolver solver(config, dataPath);

            // Run solver
            result = solver.run();
            #if defined(USE_MPI) || defined(USE_HYBRID)
            if (rank == 0)
            {
                std::cout << "Result stored in file: " << inputPath << std::endl << std::endl;
                OutputWriter::writeJsonToFile(inputPath, result);
            }
            #else
            std::cout << "Result stored in file: " << inputPath << std::endl << std::endl; 
            OutputWriter::writeJsonToFile(inputPath, result);
            #endif

        }
        else
        {
            SimulationSuite configSuite(inputPath, firstDim, reversed, ignoreConverged);

            #if defined(USE_MPI) || defined(USE_HYBRID)
            if (rank==0)
            {
                std::cout << "Starting multi run for " << configSuite.simulationDims.size() << " distinct dimensions." << std::endl << std::endl;
            }
            #else
            std::cout << "Starting multi run for " << configSuite.simulationDims.size() << " distinct dimensions." << std::endl << std::endl;
            #endif
            
            for (size_t i=0; i<configSuite.simulationDims.size(); ++i)
            {
                std::string simDim = configSuite.simulationDims[i];
                SimulationConfig config = configSuite.generateSimulation(simDim);

                #if defined(USE_MPI) || defined(USE_HYBRID)
                if (rank==0)
                {
                    std::cout << "Simulation " << i+1 << "/" << configSuite.simulationDims.size() << ":" << std::endl;
                    std::cout << "Starting simulation for D=" << simDim << "." << std::endl << std::endl;
                }
                #else
                std::cout << "Simulation " << i+1 << "/" << configSuite.simulationDims.size() << ":" << std::endl;
                std::cout << "Starting simulation for D=" << simDim << "." << std::endl << std::endl;
                #endif

                if (i==0)
                {
                    SimulationConfig config_first = configSuite.generateSimulation(firstDim);

                    if ((config.fc.empty() || config.Up.empty() || config.psic.empty()) &&
                        simDim != configSuite.firstDim && config_first.Converged)
                    {
                        config.Delta = config_first.Delta;
                        config.fc = config_first.fc;
                        config.Up = config_first.Up;
                        config.psic = config_first.psic;
                    }
                    else if (simDim != configSuite.firstDim && !config_first.Converged)
                    {
                        throw std::runtime_error("No converged solution as initial conditions available for " + simDim + "!");
                    }

                }
                else if ((config.fc.empty() || config.Up.empty() || config.psic.empty()) && i>0 && i<3)
                {
                    std::string prevDim = configSuite.simulationDims[i-1];

                    SimulationConfig config_prev = configSuite.generateSimulation(prevDim);

                    if (config_prev.Converged)
                    {
                        config.Delta = config_prev.Delta;
                        config.fc = config_prev.fc;
                        config.Up = config_prev.Up;
                        config.psic = config_prev.psic;
                    }
                    else
                    {
                        throw std::runtime_error("Previous dimension "+ prevDim +" is not converged, hence no initial data for next simulation!");
                    }
                }
                else if (config.fc.empty() || config.Up.empty() || config.psic.empty())
                {
                    std::string prevDim1 = configSuite.simulationDims[i-1];
                    std::string prevDim2 = configSuite.simulationDims[i-2];
                    std::string prevDim3 = configSuite.simulationDims[i-3];

                    SimulationConfig config_prev1 = configSuite.generateSimulation(prevDim1);
                    SimulationConfig config_prev2 = configSuite.generateSimulation(prevDim2);
                    SimulationConfig config_prev3 = configSuite.generateSimulation(prevDim3);

                    if (!config_prev1.Converged || !config_prev2.Converged || !config_prev3.Converged)
                    {
                        throw std::runtime_error("Previous dimensions not converged, hence no initial data for next simulation!");
                    }

                    // Collecting all fc
                    vec_real fc1 = config_prev1.fc;
                    vec_real fc2 = config_prev2.fc;
                    vec_real fc3 = config_prev3.fc;

                    // Collecting all Up
                    vec_real Up1 = config_prev1.Up;
                    vec_real Up2 = config_prev2.Up;
                    vec_real Up3 = config_prev3.Up;

                    // Collecting all psic
                    vec_real psic1 = config_prev1.psic;
                    vec_real psic2 = config_prev2.psic;
                    vec_real psic3 = config_prev3.psic;

                    vec_real Deltas = {config_prev1.Delta, config_prev2.Delta, config_prev3.Delta};
                    
                    // To be extrapolated values
                    vec_real fc(fc1.size(),0), Up(Up1.size(),0), psic(psic1.size(),0);
                    real_t Delta{};
                    vec_real dims = {std::stod(prevDim1), std::stod(prevDim2), std::stod(prevDim3)};
                    real_t curr_dim = std::stod(simDim);
                    
                    for (size_t j=0; j<fc1.size(); ++j)
                    {
                        vec_real y_fc = {fc1[j], fc2[j], fc3[j]};
                        vec_real y_Up = {Up1[j], Up2[j], Up3[j]};
                        vec_real y_psic = {psic1[j], psic2[j], psic3[j]};
                        
                        vec_real coeff_fc = fit_quadratic_least_squares(dims, y_fc);
                        vec_real coeff_Up = fit_quadratic_least_squares(dims, y_Up);
                        vec_real coeff_psic = fit_quadratic_least_squares(dims, y_psic);

                        fc[j] = coeff_fc[0]*curr_dim*curr_dim + coeff_fc[1]*curr_dim + coeff_fc[2];
                        Up[j] = coeff_Up[0]*curr_dim*curr_dim + coeff_Up[1]*curr_dim + coeff_Up[2]; 
                        psic[j] = coeff_psic[0]*curr_dim*curr_dim + coeff_psic[1]*curr_dim + coeff_psic[2];
                    }

                    vec_real coeff_Delta = fit_quadratic_least_squares(dims, Deltas);
                    Delta = coeff_Delta[0]*curr_dim*curr_dim + coeff_Delta[1]*curr_dim + coeff_Delta[2];
                    
                    config.Delta = Delta;
                    config.fc = fc;
                    config.Up = Up;
                    config.psic = psic;

                }

                if (ignoreConverged) config.Converged = false;
                
                NewtonSolver solver(config, dataPath);
                configSuite.multiInputDict[simDim] = solver.run();
                #if defined(USE_MPI) || defined(USE_HYBRID)
                if (rank == 0)
                {
                    std::cout << "Result stored in file: " << inputPath << std::endl;
                    OutputWriter::writeJsonToFile(inputPath, configSuite.multiInputDict);
                }
                MPI_Barrier(MPI_COMM_WORLD);
                #else
                std::cout << "Result stored in file: " << inputPath << std::endl; 
                OutputWriter::writeJsonToFile(inputPath, configSuite.multiInputDict);
                #endif
            }
        }

        #if defined(USE_MPI) || defined(USE_HYBRID)
        if (rank==0)
        {
            std::cout << "Simulation finished successfully." << std::endl;
        }
        MPI_Finalize();
        #else
        std::cout << "Simulation finished successfully." << std::endl;
        #endif

        return 0;
    }
    catch (const std::exception& e)
    {
        std::cerr << "Rank " << rank << " caught exception: " << e.what() << std::endl;

        #if defined(USE_MPI) || defined(USE_HYBRID)
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        #endif

        return 1;
    }
}