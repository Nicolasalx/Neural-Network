/*
** EPITECH PROJECT, 2024
** B-CNA-500-PAR-5-1-neuralnetwork-thibaud.cathala
** File description:
** update_parameter
*/

#include "NeuralNetwork.hpp"

#ifndef STATIC_NEURAL_NETWORK
void nn::NeuralNetwork::updateParameter(
    std::size_t inputLayer,
    std::size_t currentLayer,
    std::size_t currentNeuron,
    const double delta
)
{
    const double deltaMultlearningRate = learningRate * delta;

    for (std::size_t weightIt = 0; weightIt < neuronWeight[currentLayer].lineSize; ++weightIt) {
        neuronMomentumWeight[currentLayer](currentNeuron, weightIt) =
            (_momentum * neuronMomentumWeight[currentLayer](currentNeuron, weightIt)) +
            (deltaMultlearningRate * neuronValue[inputLayer].content[weightIt]);
        neuronWeight[currentLayer](currentNeuron, weightIt) +=
            neuronMomentumWeight[currentLayer](currentNeuron, weightIt);
    }
    neuronMomentumBias[currentLayer].content[currentNeuron] =
        (_momentum * neuronMomentumBias[currentLayer].content[currentNeuron]) + (deltaMultlearningRate);
    neuronBias[currentLayer].content[currentNeuron] += neuronMomentumBias[currentLayer].content[currentNeuron];
}

void nn::NeuralNetwork::updateLayerParameter(std::size_t inputLayer, std::size_t currentLayer)
{
    for (std::size_t i = 0; i < neuronValue[currentLayer].content.size(); ++i) {
        updateParameter(inputLayer, currentLayer, i, neuronDelta[currentLayer].content[i]);
    }
}
#endif
