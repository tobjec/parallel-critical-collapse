#pragma once

#include "common.hpp"
#include "ShootingSolver.hpp"
#include "InitialConditionGenerator.hpp"
#include "SimulationConfig.hpp"
#include "OutputWriter.hpp"

class NewtonSolver
{
    private:
        int Ntau, Nnewton;
        real_t Dim, Delta;
        real_t EpsNewton, TolNewton;
        real_t XLeft, XMid, XRight;
        int NLeft, NRight, iL, iR, iM;
        SimulationConfig config;
    
        vec_complex in0, out0, YLeft, YRight;
        vec_real fc, psic, Up, XGrid, mismatchOut;
    
        InitialConditionGenerator initGen;
        std::unique_ptr<ShootingSolver> shooter;
    
        void initializeInput();
        void generateGrid();
        void shoot(const vec_complex& inputVec, vec_complex& outputVec);
        void assembleJacobian(const vec_complex& baseInput, const vec_complex& baseOutput,
                              mat_real& jacobian);
        void solveLinearSystem(const mat_real& A, vec_real& rhs, vec_real& dx);
        real_t computeL2Norm(const vec_complex& vc);

    public:
        NewtonSolver(SimulationConfig configIn);

        void run(int maxIts);
        void writeFinalOutput();

};