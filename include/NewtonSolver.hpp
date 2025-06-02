#pragma once

#include "common.hpp"
#include "ShootingSolver.hpp"
#include "InitialConditionGenerator.hpp"
#include "SimulationConfig.hpp"
#include "OutputWriter.hpp"

class NewtonSolver
{
    private:
        SimulationConfig config;
        size_t Ntau, Nnewton, maxIts;
        real_t Dim, Delta, slowErr;
        real_t EpsNewton, TolNewton;
        real_t XLeft, XMid, XRight;
        size_t NLeft, NRight, iL, iM, iR;
    
        InitialConditionGenerator initGen;
        vec_complex YLeft, YRight, mismatchOut;
        vec_real fc, psic, Up, XGrid, in0, out0;
    
        std::unique_ptr<ShootingSolver> shooter;
    
        void initializeInput();
        void generateGrid();
        void shoot(vec_real& inputVec, vec_real& outputVec);
        
        #ifdef USE_OPENMP
        void initializeInput(InitialConditionGenerator& initGen_local, vec_complex& YLeft_local, vec_complex& YRight_local,
                             vec_real& Up_local, vec_real& psic_local, vec_real& fc_local, real_t Delta_local);
        void shoot(vec_real& inputVec, vec_real& outputVec, ShootingSolver& shooter_local,
                   InitialConditionGenerator& initGen_local, vec_complex& YLeft_local, vec_complex& YRight_local,
                   vec_complex& mismatchOut_local);
        #endif

        void assembleJacobian(const vec_real& baseInput, const vec_real& baseOutput,
                              mat_real& jacobian);
        void solveLinearSystem(const mat_real& A, vec_real& rhs, vec_real& dx);
        real_t computeL2Norm(const vec_real& vc);

    public:
        NewtonSolver(SimulationConfig configIn);

        void run();
        void writeFinalOutput();

};