/*
** EPITECH PROJECT, 2024
** B-CNA-500-PAR-5-1-neuralnetwork-thibaud.cathala
** File description:
** compute_delta
*/

#include "NeuralNetwork.hpp"

#ifndef STATIC_NEURAL_NETWORK
void nn::NeuralNetwork::computeDelta(std::size_t currentLayer, std::size_t nextLayer)
{
    double error = 0;

    for (std::size_t i = 0; i < neuronValue[currentLayer].content.size(); ++i) {
        error = 0;
        for (std::size_t j = 0; j < neuronValue[nextLayer].content.size(); ++j) {
            error += neuronDelta[nextLayer].content[j] * neuronWeight[nextLayer](j, i);
        }
        neuronDelta[currentLayer].content[i] = error * _sigmoidDerivate(neuronValue[currentLayer].content[i]);
    }
}
#endif
