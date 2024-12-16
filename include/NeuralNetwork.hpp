/*
** EPITECH PROJECT, 2024
** B-CNA-500-PAR-5-1-neuralnetwork-thibaud.cathala
** File description:
** NeuralNetwork
*/

#pragma once

#include <array>
#include <cstddef>
#include <random>
#include <vector>
#include "Matrix.hpp"
#include "TrainingData.hpp"

#ifndef STATIC_NEURAL_NETWORK
namespace nn {

class NeuralNetwork {
    public:
    struct LearningState {
        double successRate = 0;
        double accuracyRate = 0;

        LearningState &operator+=(const LearningState &other)
        {
            successRate += other.successRate;
            accuracyRate += other.accuracyRate;
            return *this;
        }

        LearningState &operator/=(double value)
        {
            successRate /= value;
            accuracyRate /= value;
            return *this;
        }
    };

    private:
    static constexpr double COOLING_RATE = 0.000'01;
    double _minLearningRate = 0.001;
    double _maxLearningRate = 10;
    double _momentum = 0.9;
    double _temperature = 1;
    std::size_t _epoch = 1;
    double _prevErrorRate = 1;
    std::mt19937 _gen{std::random_device{}()};

    template <std::size_t NbInput>
    void _setInput(const std::array<bool, NbInput> &newInput);

    double _getTemperature() const;
    double _sigmoid(double nb);
    double _sigmoidDerivate(double nb);

    void _computeNeuronValue(std::size_t layerIt, std::size_t neuronIt);

    void _createNeuron(const std::vector<std::size_t> &layerSize);
    void _createFromConf(const std::string &filename);
    void _createFromNN(const std::string &filename, int nbInput, int nbOutput);
    void _parseNeuronValueFromFile(
        const std::vector<std::string> &weight,
        const std::vector<std::string> &weightMomentum,
        const std::vector<std::string> &bias,
        const std::vector<std::string> &biasMomentum,
        const std::vector<std::size_t> &layerSize,
        const std::pair<std::size_t, std::size_t> &layerNeuronIt
    );

    public:
    std::vector<Matrix2D> neuronWeight;
    std::vector<Matrix2D> neuronMomentumWeight;
    std::vector<Matrix1D> neuronBias;
    std::vector<Matrix1D> neuronMomentumBias;
    std::vector<Matrix1D> neuronValue;
    std::vector<Matrix1D> neuronDelta;
    double learningRate = 0.1;

    NeuralNetwork(
        const std::vector<std::size_t> &layerSize,
        double learningRate,
        double minLearningRate,
        double maxLearningRate,
        double momentum = 0.9
    );

    NeuralNetwork(const std::string &filename, int nbInput, int nbOutput);

    void updateLearningRate(double currentErrorRate);

    template <std::size_t NbOutput>
    double computeMSE(const std::array<bool, NbOutput> &target) const;

    template <std::size_t NbInput>
    void computeOutput(const std::array<bool, NbInput> &input)
    {
        _setInput(input);
        for (std::size_t layerIt = 1; layerIt < neuronValue.size(); ++layerIt) {
            for (std::size_t neuronIt = 0; neuronIt < neuronValue[layerIt].content.size(); ++neuronIt) {
                _computeNeuronValue(layerIt, neuronIt);
            }
        }
    }

    template <std::size_t NbOutput>
    void computeOutputDelta(const std::array<bool, NbOutput> &target)
    {
        for (std::size_t i = 0; i < neuronValue.back().content.size(); ++i) {
            neuronDelta.back().content[i] =
                (target[i] - neuronValue.back().content[i]) * _sigmoidDerivate(neuronValue.back().content[i]);
        }
    }

    void computeDelta(std::size_t currentLayer, std::size_t nextLayer);

    void updateParameter(std::size_t inputLayer, std::size_t currentLayer, std::size_t currentNeuron, double delta);
    void updateLayerParameter(std::size_t inputLayer, std::size_t currentLayer);

    template <std::size_t NbInput, std::size_t NbOutput>
    void train(const std::array<bool, NbInput> &input, const std::array<bool, NbOutput> &target)
    {
        computeOutput(input);

        computeOutputDelta(target);
        for (std::size_t i = neuronValue.size() - 2; i > 0; --i) {
            computeDelta(i, i + 1);
        }
        for (std::size_t layerIt = 1; layerIt < neuronValue.size(); ++layerIt) {
            updateLayerParameter(layerIt - 1, layerIt);
        }
    }

    long getMaxOutputIndex() const
    {
        return std::distance(
            neuronValue.back().content.begin(),
            std::max_element(neuronValue.back().content.begin(), neuronValue.back().content.end())
        );
    }

    template <std::size_t NbInput, std::size_t NbOutput>
    LearningState computeLearningState(const std::vector<TrainingData<NbInput, NbOutput>> &evaluateData);

    void printProgress(
        double successRate,
        double accuracyRate,
        std::size_t trainingProgress,
        std::size_t batch,
        int barSize
    ) const;
    void encode(const std::string &filename) const;
};

} // namespace nn

    #include "NeuralNetwork/compute_learning_state.hpp" // NOLINT
    #include "NeuralNetwork/compute_mse.hpp"            // NOLINT
    #include "NeuralNetwork/set_input.hpp"              // NOLINT
#endif
