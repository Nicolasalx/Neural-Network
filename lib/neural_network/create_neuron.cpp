/*
** EPITECH PROJECT, 2024
** B-CNA-500-PAR-5-1-neuralnetwork-thibaud.cathala
** File description:
** create_neuron
*/

#include <format>
#include "NeuralNetwork.hpp"
#include "TrackedException.hpp"

void nn::NeuralNetwork::_createNeuron(const std::vector<std::size_t> &layerSize)
{
    if (layerSize.size() < 2) {
        throw utils::TrackedException(
            "Neural Network must contain at least 2 layer (input & output), but only " +
            std::to_string(layerSize.size()) + " was given."
        );
    }
    for (std::size_t i = 0; i < layerSize.size(); ++i) {
        if (layerSize[i] == 0) {
            throw utils::TrackedException(std::format("Invalid layer size: {} for the layer n°{}.", layerSize[i], i + 1)
            );
        }
    }

    // ? preallocat vector
    neuronValue.reserve(layerSize.size());
    neuronDelta.reserve(layerSize.size());
    neuronBias.reserve(layerSize.size());
    neuronWeight.reserve(layerSize.size());
    neuronMomentumBias.reserve(layerSize.size());
    neuronMomentumWeight.reserve(layerSize.size()); // ! value is the only usefull data for input

    // ? init input layer
    neuronValue.emplace_back(layerSize[0], false);  // maybe useless
    neuronDelta.emplace_back(0, false);             // ! useless ^
    neuronBias.emplace_back(0, false);              // ! useless ^
    neuronWeight.emplace_back(0, 0, false);         // ! useless ^
    neuronMomentumBias.emplace_back(0, false);      // ! useless ^
    neuronMomentumWeight.emplace_back(0, 0, false); // ! useless ^
    for (std::size_t i = 1; i < layerSize.size(); ++i) {
        neuronValue.emplace_back(layerSize[i], false);
        neuronDelta.emplace_back(layerSize[i], false);
        neuronBias.emplace_back(layerSize[i], true);
        neuronWeight.emplace_back(layerSize[i], layerSize[i - 1], true);
        neuronMomentumBias.emplace_back(layerSize[i], false);
        neuronMomentumWeight.emplace_back(layerSize[i], layerSize[i - 1], false);
    }
}
