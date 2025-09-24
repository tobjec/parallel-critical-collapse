#pragma once
/**
 * @file SimulationConfig.hpp
 * @brief Lightweight data structures for loading and organizing simulation parameters.
 *
 * @details
 * - **SimulationConfig**: POD-style container that initializes itself from a JSON object
 *   (or file) and exposes all parameters required by the solver stack (Newton/Shooting/IRK).
 * - **SimulationSuite**: Helper to manage a *multi*-dimension input dictionary and derive
 *   the ordered list of dimensions to run (optionally skipping already converged entries
 *   and/or reversing order with look-back seeding).
 */

#include "common.hpp"

/**
 * @struct SimulationConfig
 * @brief Single-simulation configuration (one dimension D, one Ntau, etc.).
 *
 * @section fields Key Fields
 * - `Ntau`        : Number of τ samples per period.
 * - `Dim`         : Physical dimension D.
 * - `XLeft/XMid/XRight` : Spatial match points (domain decomposition).
 * - `EpsNewton`   : Newton step damping/regularization.
 * - `PrecisionNewton` : Newton tolerance on residual.
 * - `SlowError`   : Slowly-updated error metric (diagnostics).
 * - `MaxIterNewton` : Maximum Newton iterations.
 * - `Verbose/Debug/Converged` : Execution flags and previous status.
 * - `NLeft/NRight`: Spatial grid resolution left/right of the match.
 * - `SchemeIRK`: Scheme (order) for implicit RK method (e.g. 1(1),2(4),3(6)).
 * - `PrecisionIRK`: Tolerance for implicit RK step solves.
 * - `MaxIterIRK`  : Maximum Newton iterations per IRK step.
 * - `Delta`       : Echoing period.
 * - `fc/psic/Up`  : Periodic input arrays for boundary expansions.
 */
struct SimulationConfig
{
    size_t Ntau;
    real_t Dim;
    real_t XLeft, XMid, XRight;
    real_t EpsNewton, PrecisionNewton, SlowError;
    int    MaxIterNewton;
    bool   Verbose, Debug, Converged;
    size_t NLeft, NRight;
    real_t PrecisionIRK;
    Scheme SchemeIRK;
    int    MaxIterIRK;
    real_t Delta;
    vec_real fc, psic, Up;

    /**
     * @brief Construct from a JSON object.
     *
     * Expected layout (keys must exist; some initial condition entries may be null):
     * ```
     * {
     *   "Ntau": ..., "Dim": ...,
     *   "XLeft": ..., "XMid": ..., "XRight": ...,
     *   "EpsNewton": ..., "PrecisionNewton": ..., "SlowError": ...,
     *   "MaxIterNewton": ..., "Verbose": ..., "Debug": ..., "Converged": ...,
     *   "NLeft": ..., "NRight": ...,
     *   "PrecisionIRK": ..., "SchemeIRK": ..., "MaxIterIRK": ...,
     *   "Initial_Conditions": {
     *     "Delta": <float or null>,
     *     "fc":    <array or null>,
     *     "psic":  <array or null>,
     *     "Up":    <array or null>
     *   }
     * }
     * ```
     */
    SimulationConfig(json simConfigIn)
    {
        Ntau = simConfigIn["Ntau"];
        Dim = simConfigIn["Dim"];
        XLeft = simConfigIn["XLeft"];
        XMid = simConfigIn["XMid"];
        XRight = simConfigIn["XRight"];
        EpsNewton = simConfigIn["EpsNewton"];
        PrecisionNewton = simConfigIn["PrecisionNewton"];
        SlowError = simConfigIn["SlowError"];
        MaxIterNewton = simConfigIn["MaxIterNewton"];
        Verbose = simConfigIn["Verbose"];
        Debug = simConfigIn["Debug"];
        Converged = simConfigIn["Converged"];
        NLeft = simConfigIn["NLeft"];
        NRight = simConfigIn["NRight"];
        PrecisionIRK = simConfigIn["PrecisionIRK"];
        MaxIterIRK = simConfigIn["MaxIterIRK"];

        switch (simConfigIn["SchemeIRK"].get<int>())
        {
            case 1: SchemeIRK = Scheme::IRK1; break;
            case 2: SchemeIRK = Scheme::IRK2; break;
            case 3: SchemeIRK = Scheme::IRK3; break;
            default: throw std::invalid_argument("Wrong IRK Scheme stage supplied!");
        }

        if (!simConfigIn["Initial_Conditions"]["Delta"].is_null())
        {
            Delta = simConfigIn["Initial_Conditions"]["Delta"];
        }

        if (!simConfigIn["Initial_Conditions"]["fc"].is_null())
        {
           fc = simConfigIn["Initial_Conditions"]["fc"].get<vec_real>();
        }

        if (!simConfigIn["Initial_Conditions"]["psic"].is_null())
        {
           psic = simConfigIn["Initial_Conditions"]["psic"].get<vec_real>();
        }

        if (!simConfigIn["Initial_Conditions"]["Up"].is_null())
        {
           Up = simConfigIn["Initial_Conditions"]["Up"].get<vec_real>();
        }
    }

    /**
     * @brief Load configuration from a JSON file.
     * @throws std::runtime_error if file cannot be opened.
     */
    static SimulationConfig loadFromJson(const std::string& filename)
    {
        std::ifstream inFile(filename);
        if (!inFile)
        {
            throw std::runtime_error("Could not open config file: " + filename);
        }

        json j;
        inFile >> j;

        return SimulationConfig(j);
    }

    /**
     * @brief Save a JSON object (e.g., results) to file.
     * @note Writes the JSON as-is; caller controls formatting/indentation.
     */
    static void saveToJson(const std::string& filename, json& simRes)
    {
        std::ofstream file(filename);
        file << simRes;
    }

    /// Print a human-readable configuration summary to stdout.
    void print_config()
    {
        std::cout << "Simulation configuration:" << std::endl;
        std::cout << "Ntau: " << Ntau << std::endl;
        std::cout << "Dim: " << Dim << std::endl;
        std::cout << "XLeft: " << XLeft << std::endl;
        std::cout << "XMid: " << XMid << std::endl;
        std::cout << "XRight: " << XRight << std::endl;
        std::cout << "EpsNewton: " << EpsNewton << std::endl;
        std::cout << "PrecisionNewton: " << PrecisionNewton << std::endl;
        std::cout << "SlowError: " << SlowError << std::endl;
        std::cout << "MaxIterNewton: " << MaxIterNewton << std::endl;
        std::cout << "Verbose: " << Verbose << std::endl;
        std::cout << "Debug: " << Debug << std::endl;
        std::cout << "Converged: " << Converged << std::endl;
        std::cout << "NLeft: " << NLeft << std::endl;
        std::cout << "NRight: " << NRight << std::endl;
        std::cout << "SchemeIRK: " << int(SchemeIRK)+1 << std::endl;
        std::cout << "PrecisionIRK: " << PrecisionIRK << std::endl;
        std::cout << "MaxIterIRK: " << MaxIterIRK << std::endl;
        std::cout << "Delta: " << Delta << std::endl;
        std::cout << "fc is not empty: " << !fc.empty() << std::endl;
        std::cout << "psic is not empty: " << !psic.empty() << std::endl;
        std::cout << "Up is not empty: " << !Up.empty() << std::endl;
    }
};

/**
 * @struct SimulationSuite
 * @brief Manage a set of simulations keyed by dimension (multi-D JSON input).
 *
 * @details
 * - Loads a dictionary of simulations from file (JSON with top-level keys = dimensions).
 * - Builds an ordered list of `simulationDims` to run, depending on:
 *   - `reversed`: choose ascending (`true`) or descending (`false`) order.
 *   - `ignoreConverged`: if `false`, skip entries marked `Converged=true`
 *     but *prepend* up to the last 3 converged dims as seeds (look-back).
 * - Provides `generateSimulation(dim)` to instantiate a `SimulationConfig` for a key.
 *
 * Typical JSON format:
 * ```
 * {
 *   "3.700": { ... single SimulationConfig JSON ... },
 *   "3.725": { ... },
 *   ...
 * }
 * ```
 */
struct SimulationSuite
{
    json multiInputDict;                     ///< Entire multi-simulation JSON dictionary.
    std::vector<std::string> simulationDims; ///< Ordered list of dimension keys to run.

    /**
     * @brief Construct from file path, choosing ordering and filtering.
     * @param filePath         Path to the multi-simulation JSON file.
     * @param reversed         If true, ascending order of dimensions; else descending.
     * @param ignoreConverged  If true, include all dims (even converged); else skip converged
     *                         but prepend up to 3 most recent converged as seeds.
     *
     * @throws std::filesystem::filesystem_error if file not found.
     */
    SimulationSuite(const std::string& filePath, bool reversed=false, bool ignoreConverged=false)
    {
        if (!std::filesystem::exists(filePath))
        {
            throw std::filesystem::filesystem_error("File does not exist!", filePath, std::make_error_code(std::errc::no_such_file_or_directory));
        }

        std::ifstream inputFile(filePath);
        inputFile >> multiInputDict;
        std::vector<std::string> convergedDims;

        for (const auto& dim : multiInputDict.items())
        {
            if (!ignoreConverged)
            {
                if (!dim.value()["Converged"])
                {
                    simulationDims.push_back(dim.key());
                }
                else
                {
                    convergedDims.push_back(dim.key());
                }
            }
            else
            {
                simulationDims.push_back(dim.key());
            }
        }

        if (reversed && !ignoreConverged)
        {
            std::sort(simulationDims.begin(), simulationDims.end());

            if (convergedDims.size() > 0)
            {
                std::sort(convergedDims.begin(), convergedDims.end());
                if (convergedDims.size() > 3)
                {
                    simulationDims.insert(simulationDims.begin(), convergedDims.end()-3, convergedDims.end()); 
                }
                else
                {
                    simulationDims.insert(simulationDims.begin(), convergedDims.begin(), convergedDims.end());
                }
            }
        }
        else if (!ignoreConverged)
        {
            std::sort(simulationDims.begin(), simulationDims.end(), [](auto& a, auto& b){return a>b;});

            if (convergedDims.size() > 0)
            {
                std::sort(convergedDims.begin(), convergedDims.end(), [](auto& a, auto& b){return a>b;});
                if (convergedDims.size() > 3)
                {
                    simulationDims.insert(simulationDims.begin(), convergedDims.end()-3, convergedDims.end());
                }
                else
                {
                    simulationDims.insert(simulationDims.begin(), convergedDims.begin(), convergedDims.end());
                }
            }
        }
        else if(reversed)
        {
            std::sort(simulationDims.begin(), simulationDims.end());
        }
        else
        {
            std::sort(simulationDims.begin(), simulationDims.end(), [](auto& a, auto& b){return a>b;});
        }
    }

    /**
     * @brief Create a SimulationConfig for a given dimension key.
     * @param simDimIn Dimension key as string (must exist in multiInputDict).
     * @return SimulationConfig constructed from the corresponding JSON node.
     * @throws std::out_of_range if key does not exist.
     */
    SimulationConfig generateSimulation(const std::string& simDimIn)
    {
        if (multiInputDict.contains(simDimIn))
        {
            json simConfig = multiInputDict[simDimIn];
            return SimulationConfig(simConfig);
        }
        else
        {
            throw std::out_of_range("Simulation key '" + simDimIn + "' not found in input dictionary.");
        }
    }
};
