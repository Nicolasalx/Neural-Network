/*
** EPITECH PROJECT, 2024
** B-CNA-500-PAR-5-1-neuralnetwork-thibaud.cathala
** File description:
** NeuralNetwork
*/

#include "NeuralNetwork.hpp"
#include <cmath>
#include <cstddef>
#include <vector>

nn::NeuralNetwork::NeuralNetwork(
    const std::vector<std::size_t> &layerSize,
    double learningRate,
    double minLearningRate,
    double maxLearningRate,
    double momentum
)
    : _minLearningRate(minLearningRate), _maxLearningRate(maxLearningRate), _momentum(momentum),
      learningRate(learningRate)
{
    // ! check if layer size is correct
    _createNeuron(layerSize);
}
