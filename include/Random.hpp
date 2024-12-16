/*
** EPITECH PROJECT, 2024
** B-CNA-500-PAR-5-1-neuralnetwork-thibaud.cathala
** File description:
** Random
*/

#pragma once

#include <random>

namespace nn {

class Random {
    private:
    static std::random_device rd;
    static std::mt19937 gen;
    static std::uniform_real_distribution<double> dist;

    public:
    static double normalized();
    static std::size_t range(std::size_t min, std::size_t max);
};

} // namespace nn
