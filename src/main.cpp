#include "common.hpp"
#include "SimulationConfig.hpp"
#include "NewtonSolver.hpp"

int main()
{
    try
    {
        // Load configuration
        SimulationConfig config = SimulationConfig::loadFromJson("../data/simulation.json");

        // Instantiate Newton solver with essential parameters
        NewtonSolver solver(config);

        // Run solver
        solver.run(config.MaxIterNewton);

        std::cout << "Simulation finished successfully.\n";
    }
    catch (const std::exception& ex)
    {
        std::cerr << "ERROR: " << ex.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}