#pragma once
#include "common.hpp"

class OutputWriter
{
public:
    static void writeVectorToFile(const std::string& filename, const vec_real& data);
    static void writeDelta(const std::string& filename, real_t delta);
    static void appendResultToJson(const std::string& filename, real_t dim,
                                   real_t delta,
                                   const vec_real& fc,
                                   const vec_real& psic,
                                   const vec_real& up);
};

