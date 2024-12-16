/*
** EPITECH PROJECT, 2024
** B-CNA-500-PAR-5-1-neuralnetwork-thibaud.cathala
** File description:
** update_learning_rate
*/

#include "NeuralNetwork.hpp"

void nn::NeuralNetwork::updateLearningRate(double currentErrorRate)
{
    std::uniform_real_distribution<double> dist(-0.1, 0.1);
    static constexpr double MIN_IMPROVEMENT = 0.1 / 100.0;
    double improvement = _prevErrorRate - currentErrorRate;
    _temperature = _getTemperature();

    if (improvement > 0) {
        if (improvement < MIN_IMPROVEMENT) {
            learningRate = std::min(_maxLearningRate, learningRate * 1.02);
        } else {
            learningRate = std::min(_maxLearningRate, learningRate * 1.05);
        }
    } else {
        learningRate = std::max(_minLearningRate, learningRate / 2.0);
        learningRate =
            std::clamp(learningRate * (1.0 + (dist(_gen) * _temperature)), _minLearningRate, _maxLearningRate);
    }
    _prevErrorRate = currentErrorRate;
    ++_epoch;
}
