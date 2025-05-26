#include "OutputWriter.hpp"

void OutputWriter::writeVectorToFile(const std::string& filename, const vec_real& data)
{
    std::ofstream out(filename);
    if (!out)
    {
        throw std::runtime_error("Could not open file for writing: " + filename);
    }

    out.precision(16);
    for (const auto& val : data)
    {
        out << val << "\n";
    }
}

void OutputWriter::writeDelta(const std::string& filename, real_t delta)
{
    std::ofstream out(filename);
    if (!out)
    {
        throw std::runtime_error("Could not open Delta file for writing: " + filename);
    }

    out.precision(16);
    out << delta << "\n";
}

void OutputWriter::appendResultToJson(const std::string& filename, real_t dim,
    real_t delta,
    const vec_real& fc,
    const vec_real& psic,
    const vec_real& up)
{
    json j;

    // Load existing results if file exists
    if (std::filesystem::exists(filename))
    {
    std::ifstream inFile(filename);
    inFile >> j;
    }

    // Convert everything to JSON
    std::string key = std::to_string(dim);
    j[key] = {
    {"Delta", delta},
    {"fc", fc},
    {"psic", psic},
    {"Up", up}
    };

    // Write back
    std::ofstream outFile(filename);
    outFile << std::setw(2) << j << std::endl;
}