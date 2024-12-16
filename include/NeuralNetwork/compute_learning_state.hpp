/*
** EPITECH PROJECT, 2024
** B-CNA-500-PAR-5-1-neuralnetwork-thibaud.cathala
** File description:
** compute_learning_state
*/

#pragma once

#include "NeuralNetwork.hpp"

namespace nn {

template <std::size_t NbInput, std::size_t NbOutput>
NeuralNetwork::LearningState NeuralNetwork::computeLearningState(
    const std::vector<TrainingData<NbInput, NbOutput>> &evaluateData
)
{
    LearningState learningState;

    for (const auto &it : evaluateData) {
        computeOutput(it.first);
        learningState.accuracyRate += computeMSE(it.second);
        if (std::distance(
                neuronValue.back().content.begin(),
                std::max_element(neuronValue.back().content.begin(), neuronValue.back().content.end())
            ) == std::distance(it.second.begin(), std::max_element(it.second.begin(), it.second.end()))) {
            learningState.successRate += 1;
        }
    }
    learningState.accuracyRate /= static_cast<double>(evaluateData.size());
    learningState.successRate /= static_cast<double>(evaluateData.size());

    learningState.accuracyRate = (1 - learningState.accuracyRate);
    return learningState;
}

} // namespace nn
