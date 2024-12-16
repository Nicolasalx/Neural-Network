/*
** EPITECH PROJECT, 2024
** B-CNA-500-PAR-5-1-neuralnetwork-thibaud.cathala
** File description:
** Matrix
*/

#pragma once

#include <algorithm>
#ifdef STATIC_NEURAL_NETWORK
    #include <array>
#else
    #include <vector>
#endif
#include "Random.hpp"

namespace nn {

#ifdef STATIC_NEURAL_NETWORK
struct Matrix1D {
    private:
    std::vector<double> _content;

    public:
    Matrix1D() = default;

    void create(std::size_t nbLine, bool randomInit)
    {
        if (nbLine == 0) {
            return;
        }
        _content.resize(nbLine, 0);
        if (randomInit) {
            std::generate(_content.begin(), _content.end(), Random::normalized);
        }
    }

    double &operator[](std::size_t index)
    {
        return _content[index];
    }

    const double &operator[](std::size_t index) const
    {
        return _content[index];
    }
};
#else
struct Matrix1D {
    std::vector<double> content;

    Matrix1D(std::size_t nbLine, bool randomInit) : content(nbLine, 0)
    {
        if (randomInit) {
            std::generate(content.begin(), content.end(), Random::normalized);
        }
    }
};
#endif

#ifdef STATIC_NEURAL_NETWORK
struct Matrix2D {
    private:
    std::size_t _lineSize = 0;
    std::vector<double> _content;

    public:
    Matrix2D() = default;

    void create(std::size_t nbLine, std::size_t lineSize, bool randomInit)
    {
        if (nbLine == 0 || lineSize == 0) {
            return;
        }
        _lineSize = lineSize;
        _content.resize(nbLine * lineSize, 0);
        if (randomInit) {
            std::generate(_content.begin(), _content.end(), Random::normalized);
        }
    }

    double &operator()(std::size_t y, std::size_t x)
    {
        return _content[y * _lineSize + x];
    }

    const double &operator()(std::size_t y, std::size_t x) const
    {
        return _content[y * _lineSize + x];
    }
};
#else
struct Matrix2D {
    public:
    std::size_t nbLine = 0;
    std::size_t lineSize = 0;

    private:
    std::vector<double> _content;

    public:
    Matrix2D(std::size_t nbLine, std::size_t lineSize, bool randomInit)
        : nbLine(nbLine), lineSize(lineSize), _content(nbLine * lineSize, 0)
    {
        if (randomInit) {
            std::generate(_content.begin(), _content.end(), Random::normalized);
        }
    }

    double &operator()(std::size_t y, std::size_t x)
    {
        return _content[y * lineSize + x];
    }

    const double &operator()(std::size_t y, std::size_t x) const
    {
        return _content[y * lineSize + x];
    }
};
#endif

} // namespace nn
