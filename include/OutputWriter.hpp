#pragma once
/**
 * @file OutputWriter.hpp
 * @brief Static utility class for writing simulation results to files.
 *
 * @details
 * Provides simple convenience wrappers to persist simulation data
 * (vectors, matrices, JSON dictionaries) to disk in human-readable
 * text or JSON format. All functions are static and can be called
 * without instantiating the class.
 */

#include "common.hpp"

/**
 * @class OutputWriter
 * @brief Collection of static methods for file output.
 *
 * @section usage Usage
 * - Call OutputWriter::writeVector("file.txt", vec) to dump data.
 * - Overloaded for real and complex vectors.
 * - JSON objects are written with indentation for readability.
 */
class OutputWriter
{
  public:
    /**
     * @brief Write a vector of reals to a text file.
     * @param filename Path to output file.
     * @param data     Real-valued vector.
     *
     * @details
     * Values are written line by line with full precision.
     */
    static void writeVector(const std::string& filename, const vec_real& data);

    /**
     * @brief Write a vector of complex numbers to a text file.
     * @param filename Path to output file.
     * @param data     Complex-valued vector.
     *
     * @details
     * Each line contains "real imag".
     */
    static void writeVector(const std::string& filename, const vec_complex& data);

    /**
     * @brief Write a matrix of reals to a text file.
     * @param filename Path to output file.
     * @param data     2D matrix of real values.
     *
     * @details
     * Each row is written on one line, values separated by spaces.
     */
    static void writeMatrix(const std::string& filename, const mat_real& data);

    /**
     * @brief Write a JSON dictionary to a file.
     * @param filename Path to output file.
     * @param dictionary JSON object.
     *
     * @details
     * JSON is written with indentation and UTF-8 encoding.
     */
    static void writeJsonToFile(const std::string& filename, json& dictionary);
};
