#include "common.hpp"
#include "SimulationConfig.hpp"
#include "NewtonSolver.hpp"

int main(int argc, char* argv[])
{

    #ifdef USE_MPI
    MPI_Init(&argc, &argv);
    #endif

    std::string input_path{};

    if (argc<2)
    {
        input_path = "/home/tjechtl/Documents/Education/TUW/Master_Thesis/parallel-critical-collapse/data/simulation.json";
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

    #ifdef USE_MPI
    MPI_Finalize();
    #endif

    std::cout << "Simulation finished successfully.\n";

    return EXIT_SUCCESS;
}