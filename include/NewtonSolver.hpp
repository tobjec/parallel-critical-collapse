#pragma once

#include "common.hpp"
#include "ShootingSolver.hpp"
#include "InitialConditionGenerator.hpp"

class NewtonSolver
{
    private:
        int ny, n3;
        real_t Dim;
        real_t Delta;
        real_t EpsNewton;
        real_t TolNewton;
    
        vec_real in0, out0, change;
    
        InitialConditionGenerator initGen;
        std::unique_ptr<ShootingSolver> shooter;
    
        void initializeInput();
        void shoot(const vec_real& input, vec_real& output);
        void assembleJacobian(const vec_real& baseInput,
                              mat_real& jacobian);
        void solveLinearSystem(const mat_real& A, vec_real& rhs, vec_real& dx);
        real_t computeL2Norm(const vec_real& v);

    public:
        NewtonSolver(int numGridPts, real_t dim, real_t delta,
                    real_t epsNewton, real_t tolNewton);

        void run(int maxIts);
        void writeFinalOutput();

};