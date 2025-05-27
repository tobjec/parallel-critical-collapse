#pragma once
#include "common.hpp"

struct SimulationConfig
{
    size_t Ntau;
    real_t Dim;
    real_t XLeft, XMid, XRight;
    real_t EpsNewton, PrecisionNewton, SlowError;
    int OutEvery, MaxIterNewton;
    bool Verbose, UseLogGrid, Debug;
    size_t NLeft, NRight;
    real_t Tolerance;
    int TimeStep;
    real_t PrecisionIRK;
    int MaxIterIRK;
    real_t Delta;
    vec_real fc, psic, Up;

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
            j["Ntau"],
            j["Dim"],
            j["XLeft"], j["XMid"], j["XRight"],
            j["EpsNewton"], j["PrecisionNewton"], j["SlowError"],
            j["OutEvery"], j["MaxIterNewton"],
            j["Verbose"], j["UseLogGrid"], j["Debug"],
            j["NLeft"], j["NRight"],
            j["Tolerance"],
            j["TimeStep"],
            j["PrecisionIRK"], j["MaxIterIRK"], j["Initial_Conditions"]["Delta"],
            j["Initial_Conditions"]["fc"], j["Initial_Conditions"]["psic"],
            j["Initial_Conditions"]["Up"]
        };
    }
};
