/*
** EPITECH PROJECT, 2024
** B-CNA-500-PAR-5-1-neuralnetwork-thibaud.cathala
** File description:
** encode
*/

#include <fstream>
#include "NeuralNetwork.hpp"
#include "TrackedException.hpp"

void nn::NeuralNetwork::encode(const std::string &filename) const
{
    std::ofstream file(filename);

    if (!file.is_open()) {
        throw utils::TrackedException("Failed to open file: " + filename + '.');
    }
    file << "learning_rate: " << learningRate << '\n';
    file << "min_learning_rate: " << _minLearningRate << '\n';
    file << "max_learning_rate: " << _maxLearningRate << '\n';
    file << "momentum: " << _momentum << '\n';
    file << "nb_layer: " << neuronValue.size() << '\n';
    for (std::size_t i = 0; i < neuronValue.size(); ++i) {
        file << "nb_neuron: " << neuronValue[i].content.size() << '\n';
        if (i == 0) {
            continue;
        }
        for (std::size_t j = 0; j < neuronValue[i].content.size(); ++j) {
            file << "    nb_weight: " << neuronWeight[i].lineSize;
            for (std::size_t weightIt = 0; weightIt < neuronWeight[i].lineSize; ++weightIt) {
                file << ", " << neuronWeight[i](j, weightIt);
            }
            file << '\n';
            file << "    nb_weight_momentum: " << neuronMomentumWeight[i].lineSize;
            for (std::size_t weightIt = 0; weightIt < neuronMomentumWeight[i].lineSize; ++weightIt) {
                file << ", " << neuronMomentumWeight[i](j, weightIt);
            }
            file << '\n';
            file << "    bias: " << neuronBias[i].content[j] << '\n';
            file << "    bias_momentum: " << neuronMomentumBias[i].content[j] << '\n';
        }
        file << '\n';
    }
}
