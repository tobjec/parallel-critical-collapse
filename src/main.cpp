#include "common.hpp"
#include "SimulationConfig.hpp"
#include "NewtonSolver.hpp"

int main(int argc, char* argv[])
{

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

    std::string input_path{};

    if (argc<2)
    {
        input_path = "/home/tjechtl/Documents/Education/TUW/Master_Thesis/parallel-critical-collapse/data/simulation_4D.json";
    }
    else
    {
        input_path = std::string(argv[1]);
    }

    // Load configuration
    SimulationConfig config = SimulationConfig::loadFromJson(input_path);

    // Instantiate Newton solver with essential parameters
    NewtonSolver solver(config);

    // Run solver
    solver.run();

    #if defined(USE_MPI) || defined(USE_HYBRID)
    MPI_Finalize();
    #endif

    std::cout << "Simulation finished successfully.\n";

    return EXIT_SUCCESS;
}