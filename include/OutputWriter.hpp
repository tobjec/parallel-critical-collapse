#pragma once
#include "common.hpp"

class OutputWriter
{
    public:
        static void writeVector(const std::string& filename, const vec_real& data);
        static void writeVector(const std::string& filename, const vec_complex& data);
        static void writeMatrix(const std::string& filename, const mat_real& data);
        static void writeJsonToFile(const std::string& filename, json& dictionary);
};

