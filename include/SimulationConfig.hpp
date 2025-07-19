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
        std::cout << "PrecisionIRK: " << PrecisionIRK << std::endl;
        std::cout << "MaxIterIRK: " << MaxIterIRK << std::endl;
        std::cout << "Delta: " << Delta << std::endl;
        std::cout << "fc is not empty: " << !fc.empty() << std::endl;
        std::cout << "psic is not empty: " << !psic.empty() << std::endl;
        std::cout << "Up is not empty: " << !Up.empty() << std::endl;
    }
};

struct SimulationSuite
{
    json multiInputDict;
    std::vector<std::string> simulationDims;

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
                else if (!reversed)
                {
                    convergedDims.push_back(dim.key());
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
