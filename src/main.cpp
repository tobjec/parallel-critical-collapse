#include "common.hpp"
#include "SimulationConfig.hpp"
#include "NewtonSolver.hpp"

int main(int argc, char** argv)
{
    try
    {
        // Load configuration
        SimulationConfig config = SimulationConfig::loadFromJson("data/simulation.json");

        // Instantiate Newton solver with essential parameters
        NewtonSolver solver(
            config.Ny,
            config.Dim,
            config.XMid,     
            config.EpsNewton,
            config.PrecisionNewton
        );

        // Run solver
        solver.run(50);

        std::cout << "Simulation finished successfully.\n";
    }
    catch (const std::exception& ex)
    {
        std::cerr << "ERROR: " << ex.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}