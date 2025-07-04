#pragma once
#include "common.hpp"

struct SimulationConfig
{
    size_t Ntau;
    real_t Dim;
    real_t XLeft, XMid, XRight;
    real_t EpsNewton, PrecisionNewton, SlowError;
    int MaxIterNewton;
    bool Verbose, Debug, Converged;
    size_t NLeft, NRight;
    real_t PrecisionIRK;
    int MaxIterIRK;
    real_t Delta;
    vec_real fc, psic, Up;

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
        Delta = simConfigIn["Initial_Conditions"]["Delta"];
        fc = simConfigIn["Initial_Conditions"]["fc"].get<std::vector<real_t>>();
        psic = simConfigIn["Initial_Conditions"]["psic"].get<std::vector<real_t>>();
        Up = simConfigIn["Initial_Conditions"]["Up"].get<std::vector<real_t>>();
    }

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

    static void saveToJson(const std::string& filename, json& simRes)
    {
        std::ofstream file(filename);
        file << simRes;

    }
};

struct SimulationSuite
{
    json multiInputDict;
    std::string firstDim;
    std::vector<std::string> simulationDims;

    SimulationSuite(const std::string& filePath, std::string firstDimIn="4.000", bool reversed=false)
    {
        if (!std::filesystem::exists(filePath))
        {
            throw std::filesystem::filesystem_error("File does not exist!", filePath, std::make_error_code(std::errc::no_such_file_or_directory));
        }

        firstDim = firstDimIn;

        std::ifstream inputFile(filePath);
        inputFile >> multiInputDict;

        for (const auto& dim : multiInputDict.items())
        {
            if (!dim.value()["Converged"])
            {
                simulationDims.push_back(dim.key());
            }
        }
        
        if (reversed)
        {
            std::sort(simulationDims.begin(), simulationDims.end());
        }
        else
        {
            std::sort(simulationDims.begin(), simulationDims.end(), [](auto& a, auto& b){return a>b;});
        }

    }

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
