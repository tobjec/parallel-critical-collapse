#include "common.hpp"
#include "SimulationConfig.hpp"
#include "NewtonSolver.hpp"

int main()
{
    try
    {
        // Load configuration
        SimulationConfig config = SimulationConfig::loadFromJson("/home/tjechtl/Documents/Education/TUW/Master_Thesis/parallel-critical-collapse/data/simulation.json");

        // Instantiate Newton solver with essential parameters
        NewtonSolver solver(config);

        // Run solver
        solver.run();

        std::cout << "Simulation finished successfully.\n";
    }
    catch (const std::exception& ex)
    {
        std::cerr << "ERROR: " << ex.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}