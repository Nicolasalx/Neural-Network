/*
** EPITECH PROJECT, 2024
** B-CNA-500-PAR-5-1-neuralnetwork-thibaud.cathala
** File description:
** compute_mse
*/

#pragma once

#include "NeuralNetwork.hpp"

namespace nn {

template <std::size_t NbOutput>
double NeuralNetwork::computeMSE(const std::array<bool, NbOutput> &target) const
{
    double error = 0;
    double diff = 0;

    for (std::size_t i = 0; i < neuronValue.back().content.size(); ++i) {
        diff = neuronValue.back().content[i] - target[i];
        error += diff * diff;
    }
    return error / static_cast<double>(neuronValue.back().content.size());
}

} // namespace nn
