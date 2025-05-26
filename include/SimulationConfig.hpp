#pragma once
#include "common.hpp"

struct SimulationConfig
{
    int Ny;
    real_t Dim;
    real_t XLeft, XMid, XRight;
    real_t EpsNewton, PrecisionNewton, SlowError;
    int OutEvery;
    bool Verbose, UseLogGrid, Debug;
    int NLeft, NRight;
    real_t Tolerance;
    int TimeStep;
    real_t PrecisionIRK;

    static SimulationConfig loadFromJson(const std::string& filename)
    {
        std::ifstream inFile(filename);
        if (!inFile)
        {
            throw std::runtime_error("Could not open config file: " + filename);
        }

        json j;
        inFile >> j;

        return {
            j["Ny"],
            j["Dim"],
            j["XLeft"], j["XMid"], j["XRight"],
            j["EpsNewton"], j["PrecisionNewton"], j["SlowError"],
            j["OutEvery"],
            j["Verbose"], j["UseLogGrid"], j["Debug"],
            j["NLeft"], j["NRight"],
            j["Tolerance"],
            j["TimeStep"],
            j["PrecisionIRK"]
        };
    }
};
