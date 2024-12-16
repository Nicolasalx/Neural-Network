/*
** EPITECH PROJECT, 2024
** B-CNA-500-PAR-5-1-neuralnetwork-thibaud.cathala
** File description:
** helper_formula
*/

#include "NeuralNetwork.hpp"

#ifndef STATIC_NEURAL_NETWORK
double nn::NeuralNetwork::_getTemperature() const
{
    return std::max(0.01, 1.0 / (1.0 + COOLING_RATE * static_cast<double>(_epoch)));
}

double nn::NeuralNetwork::_sigmoid(double nb)
{
    return 1.0 / (1.0 + std::exp(-nb));
}

double nn::NeuralNetwork::_sigmoidDerivate(double nb)
{
    return nb * (1.0 - nb);
}
#endif
