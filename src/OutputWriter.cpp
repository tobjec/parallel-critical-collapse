#include "OutputWriter.hpp"

void OutputWriter::writeVector(const std::string& filename, const vec_real& data)
{
    std::ofstream outfile(filename);
    if (!outfile)
    {
        throw std::runtime_error("Could not open file for writing: " + filename);
    }

    outfile << std::setprecision(16);
    for (size_t i=0; i<data.size()-1; ++i)
    {
        outfile << data[i] << std::endl;
    }
    outfile << data[data.size()-1] << std::endl;
}

void OutputWriter::writeVector(const std::string& filename, const vec_complex& data)
{
    std::ofstream outfile(filename);
    if (!outfile)
    {
        throw std::runtime_error("Could not open file for writing: " + filename);
    }

    outfile << std::setprecision(16);
    for (size_t i=0; i<data.size()-1; ++i)
    {
        outfile <<"(" << data[i].real() << ", " << data[i].imag() << "), " << std::endl;
    }
    outfile <<"(" << data[data.size()-1].real() << ", " << data[data.size()-1].imag() << ")" << std::endl;
}

void OutputWriter::writeMatrix(const std::string& filename, const mat_real& data)
{
    std::ofstream outfile(filename, std::ios::out);
    if (!outfile)
    {
        throw std::runtime_error("Could not open file for writing: " + filename);
    }

    outfile << std::setprecision(16);
    for (size_t i=0; i<data.size(); ++i)
    {
        for (size_t j=0; j<data[i].size()-1; ++j)
        {
            outfile << data[i][j] << ", ";
        }
        outfile << data[i][data[i].size()-1] << std::endl;
    }
}

void OutputWriter::writeJsonToFile(const std::string& filename, json& dictionary)
{
    std::ofstream outputfile(filename);
    if (!outputfile)
    {
        throw std::runtime_error("Could not open file for writing: " + filename);
    }
    outputfile << dictionary;
}