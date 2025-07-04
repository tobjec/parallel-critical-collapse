#include "common.hpp"
#include "SimulationConfig.hpp"
#include "NewtonSolver.hpp"

int main(int argc, char* argv[])
{

    bool singleRun = true;
    bool ignoreConverged = false;
    std::string inputPath{};
    std::string firstDim{"4.000"};

    if (argc<2)
    {
        inputPath = "/home/tjechtl/Documents/Education/TUW/Master_Thesis/parallel-critical-collapse/data/simulation_4D.json";
    }
    else
    {

        for (size_t i=1; i<argc; ++i)
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
            else if (arg == "--ignore-convergence")
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
        }

    }

    #if defined(USE_MPI)
    MPI_Init(&argc, &argv);
    #elif defined(USE_HYBRID)
    int required = MPI_THREAD_FUNNELED;
    int provided = -1;
    MPI_Init_thread(&argc, &argv, required, &provided);
    
    if (provided < required)
    {
        std::cout << "Not enough thread support!" << std::endl;
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    } 
    #endif

    #if defined(USE_MPI) || defined(USE_HYBRID)
    int rank, size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    #endif

    if (singleRun)
    {
        json result;
        // Load configuration
        SimulationConfig config = SimulationConfig::loadFromJson(inputPath);

        // Instantiate Newton solver with essential parameters
        NewtonSolver solver(config);

        // Run solver
        result = solver.run();
        #if defined(USE_MPI) || defined(USE_HYBRID)
        if (rank == 0)
        {
            OutputWriter::writeJsonToFile(inputPath, result);
        }
        #else 
        OutputWriter::writeJsonToFile(inputPath, result);
        #endif

    }
    else
    {
        SimulationSuite configSuite(inputPath, firstDim);
        
        for (size_t i=0; i<configSuite.simulationDims.size(); ++i)
        {
            std::string simDim = configSuite.simulationDims[i];
            SimulationConfig config = configSuite.generateSimulation(simDim);
            
            if (i==0)
            {
                if ((config.fc.empty() || config.Up.empty() || config.psic.empty()) &&
                    simDim != configSuite.firstDim && configSuite.multiInputDict[firstDim]["Converged"])
                {
                    config.Delta = configSuite.multiInputDict[firstDim]["Initial_Condition"]["Delta"];
                    config.fc = configSuite.multiInputDict[firstDim]["Initial_Condition"]["fc"].get<std::vector<real_t>>();
                    config.Up = configSuite.multiInputDict[firstDim]["Initial_Condition"]["Up"].get<std::vector<real_t>>();
                    config.psic = configSuite.multiInputDict[firstDim]["Initial_Condition"]["psic"].get<std::vector<real_t>>();
                }
                else
                {
                    throw std::runtime_error("No converged solution as initial conditions available for " + simDim + "!");
                }
            }
            else if ((config.fc.empty() || config.Up.empty() || config.psic.empty()) && i>0 && i<3)
            {
                std::string prevDim = configSuite.simulationDims[i-1];

                if (configSuite.multiInputDict[prevDim]["Converged"])
                {
                    config.Delta = configSuite.multiInputDict[prevDim]["Initial_Condition"]["Delta"];
                    config.fc = configSuite.multiInputDict[prevDim]["Initial_Condition"]["fc"].get<std::vector<real_t>>();
                    config.Up = configSuite.multiInputDict[prevDim]["Initial_Condition"]["Up"].get<std::vector<real_t>>();
                    config.psic = configSuite.multiInputDict[prevDim]["Initial_Condition"]["psic"].get<std::vector<real_t>>();
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

                if (!configSuite.multiInputDict[prevDim1]["Converged"] || !configSuite.multiInputDict[prevDim2]["Converged"]
                    || !configSuite.multiInputDict[prevDim3]["Converged"])
                {
                    throw std::runtime_error("Previous dimensions not converged, hence no initial data for next simulation!");
                }

                // Collecting all fc
                vec_real fc1 = configSuite.multiInputDict[prevDim1]["Initial_Condition"]["fc"].get<std::vector<real_t>>();
                vec_real fc2 = configSuite.multiInputDict[prevDim2]["Initial_Condition"]["fc"].get<std::vector<real_t>>();
                vec_real fc3 = configSuite.multiInputDict[prevDim3]["Initial_Condition"]["fc"].get<std::vector<real_t>>();

                // Collecting all Up
                vec_real Up1 = configSuite.multiInputDict[prevDim1]["Initial_Condition"]["Up"].get<std::vector<real_t>>();
                vec_real Up2 = configSuite.multiInputDict[prevDim2]["Initial_Condition"]["Up"].get<std::vector<real_t>>();
                vec_real Up3 = configSuite.multiInputDict[prevDim3]["Initial_Condition"]["Up"].get<std::vector<real_t>>();

                // Collecting all psic
                vec_real psic1 = configSuite.multiInputDict[prevDim1]["Initial_Condition"]["psic"].get<std::vector<real_t>>();
                vec_real psic2 = configSuite.multiInputDict[prevDim2]["Initial_Condition"]["psic"].get<std::vector<real_t>>();
                vec_real psic3 = configSuite.multiInputDict[prevDim3]["Initial_Condition"]["psic"].get<std::vector<real_t>>();

                vec_real Deltas = {configSuite.multiInputDict[prevDim1]["Initial_Condition"]["Delta"],
                                   configSuite.multiInputDict[prevDim2]["Initial_Condition"]["Delta"],
                                   configSuite.multiInputDict[prevDim3]["Initial_Condition"]["Delta"]};
                
                // To be extrapolated values
                vec_real fc(fc1.size(),0), Up(Up1.size(),0), psic(psic1.size(),0);
                real_t Delta{};
                vec_real dims = {std::stod(prevDim1), std::stod(prevDim2), std::stod(prevDim3)};
                real_t curr_dim = std::stod(simDim);
                
                for (size_t i=0; i<fc1.size(); ++i)
                {
                    vec_real y_fc = {fc1[i], fc2[i], fc3[i]};
                    vec_real y_Up = {Up1[i], Up2[i], Up3[i]};
                    vec_real y_psic = {psic1[i], psic2[i], psic3[i]};
                    
                    vec_real coeff_fc = fit_quadratic_least_squares(dims, y_fc);
                    vec_real coeff_Up = fit_quadratic_least_squares(dims, y_Up);
                    vec_real coeff_psic = fit_quadratic_least_squares(dims, y_psic);

                    fc[i] = coeff_fc[0]*curr_dim*curr_dim + coeff_fc[1]*curr_dim + coeff_fc[2];
                    Up[i] = coeff_Up[0]*curr_dim*curr_dim + coeff_Up[1]*curr_dim + coeff_Up[2]; 
                    psic[i] = coeff_psic[0]*curr_dim*curr_dim + coeff_psic[1]*curr_dim + coeff_psic[2];
                }

                vec_real coeff_Delta = fit_quadratic_least_squares(dims, Deltas);
                Delta = coeff_Delta[0]*curr_dim*curr_dim + coeff_Delta[1]*curr_dim + coeff_Delta[2];
                
                config.Delta = Delta;
                config.fc = fc;
                config.Up = Up;
                config.psic = psic;

            }
            
            NewtonSolver solver(config);
            configSuite.multiInputDict[simDim] = solver.run();
            #if defined(USE_MPI) || defined(USE_HYBRID)
            if (rank == 0)
            {
                OutputWriter::writeJsonToFile(inputPath, configSuite.multiInputDict);
            }
            #else 
            OutputWriter::writeJsonToFile(inputPath, configSuite.multiInputDict);
            #endif
        }
    }

    

    #if defined(USE_MPI) || defined(USE_HYBRID)
    MPI_Finalize();
    #endif

    std::cout << "Simulation finished successfully.\n";

    return EXIT_SUCCESS;
}