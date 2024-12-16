/*
** EPITECH PROJECT, 2024
** B-CNA-500-PAR-5-1-neuralnetwork-thibaud.cathala
** File description:
** StaticNeuralNetwork
*/

#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <fstream>
#include <immintrin.h>
#include <iomanip>
#include <iostream>
#include <random>
#include "Random.hpp"
#include "TrackedException.hpp"
#include "TrainingData.hpp"

#ifdef STATIC_NEURAL_NETWORK
namespace nn {

template <std::size_t NbLayer, std::array<std::size_t, NbLayer> LayerSize>
class Layer2D {
    private:
    static constexpr std::array<size_t, NbLayer> computeOffset()
    {
        std::array<size_t, NbLayer> offset;
        std::size_t current = 0;

        for (std::size_t i = 0; i < NbLayer; i++) {
            offset[i] = current;
            current += LayerSize[i];
        }
        return offset;
    }

    static constexpr std::array<std::size_t, NbLayer> LAYER_OFFSET = computeOffset();
    static constexpr std::size_t TOTAL_SIZE = LAYER_OFFSET.back() + LayerSize.back();

    std::array<double, TOTAL_SIZE> _data;

    public:
    Layer2D() = default;

    void fill(double value)
    {
        std::fill(_data.begin(), _data.end(), value);
    }

    void fillRandomNormalized()
    {
        std::generate(_data.begin(), _data.end(), []() { return Random::normalized(); });
    }

    double &operator()(size_t layerIt, size_t neuronIt)
    {
        return _data[LAYER_OFFSET[layerIt] + neuronIt];
    }

    const double &operator()(size_t layerIt, size_t neuronIt) const
    {
        return _data[LAYER_OFFSET[layerIt] + neuronIt];
    }

    double &back(size_t neuronIt)
    {
        return _data[LAYER_OFFSET[NbLayer - 1] + neuronIt];
    }

    const double &back(size_t neuronIt) const
    {
        return _data[LAYER_OFFSET[NbLayer - 1] + neuronIt];
    }

    double &front(size_t neuronIt)
    {
        return _data[LAYER_OFFSET[0] + neuronIt];
    }

    const double &front(size_t neuronIt) const
    {
        return _data[LAYER_OFFSET[0] + neuronIt];
    }

    static constexpr std::size_t getTotalSize()
    {
        return TOTAL_SIZE;
    }

    static constexpr std::size_t getLayerSize(size_t layer)
    {
        return LayerSize[layer];
    }

    static constexpr std::size_t getLayerOffset(size_t layer)
    {
        return LAYER_OFFSET[layer];
    }
};

template <std::size_t NbLayer, std::array<std::size_t, NbLayer> LayerSize>
class Layer3D {
    private:
    static constexpr std::pair<std::array<size_t, NbLayer>, std::size_t> computeOffset()
    {
        std::array<size_t, NbLayer> offset;
        std::size_t current = 0;

        offset[0] = 0;
        for (std::size_t i = 1; i < NbLayer; i++) {
            offset[i] = current;
            current += LayerSize[i] * LayerSize[i - 1];
        }
        return std::make_pair(offset, current);
    }

    static constexpr auto OFFSET_DATA = computeOffset();
    static constexpr auto LAYER_OFFSETS = OFFSET_DATA.first;
    static constexpr auto TOTAL_SIZE = OFFSET_DATA.second;

    std::array<double, TOTAL_SIZE> _data;

    public:
    Layer3D() = default;

    void fill(double value)
    {
        std::fill(_data.begin(), _data.end(), value);
    }

    void fillRandomNormalized()
    {
        std::generate(_data.begin(), _data.end(), Random::normalized);
    }

    double &operator()(size_t layerIt, size_t neuronIt, size_t weightIt)
    {
        return _data[LAYER_OFFSETS[layerIt] + neuronIt * LayerSize[layerIt - 1] + weightIt];
    }

    const double &operator()(size_t layerIt, size_t neuronIt, size_t weightIt) const
    {
        return _data[LAYER_OFFSETS[layerIt] + neuronIt * LayerSize[layerIt - 1] + weightIt];
    }

    static constexpr size_t getTotalSize()
    {
        return TOTAL_SIZE;
    }

    static constexpr size_t getLayerOffset(size_t layer)
    {
        return LAYER_OFFSETS[layer];
    }

    static constexpr size_t getPrevLayerSize(size_t layer)
    {
        return LayerSize[layer - 1];
    }
};

template <std::size_t NbLayer, std::array<std::size_t, NbLayer> LayerSize>
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

    Layer3D<NbLayer, LayerSize> neuronWeight;
    Layer3D<NbLayer, LayerSize> neuronMomentumWeight;
    Layer2D<NbLayer, LayerSize> neuronBias;
    Layer2D<NbLayer, LayerSize> neuronMomentumBias;
    Layer2D<NbLayer, LayerSize> neuronValue;
    Layer2D<NbLayer, LayerSize> neuronDelta;
    double learningRate = 0.1;

    private:
    static constexpr double COOLING_RATE = 0.000'01;
    double _minLearningRate = 0.001;
    double _maxLearningRate = 10;
    double _momentum = 0.9;
    double _temperature = 1;
    std::size_t _epoch = 1;
    double _prevErrorRate = 1;
    std::mt19937 _gen{std::random_device{}()};

    void _setInput(const std::array<double, LayerSize.front()> &newInput)
    {
        for (std::size_t i = 0; i < newInput.size(); ++i) {
            neuronValue.front(i) = newInput[i];
        }
    }

    double _getTemperature() const
    {
        return std::max(0.01, 1.0 / (1.0 + COOLING_RATE * static_cast<double>(_epoch)));
    }

    double _sigmoid(double nb)
    {
        return 1.0 / (1.0 + std::exp(-nb));
    }

    double _sigmoidDerivate(double nb)
    {
        return nb * (1.0 - nb);
    }

    void _computeNeuronValue(std::size_t layerIt, std::size_t neuronIt)
    {
    #if (defined(__AVX512F__) && !defined(CHECK_MEMORY))
        double sigma = neuronBias(layerIt, neuronIt);
        constexpr std::size_t VECTOR_SIZE = sizeof(__m512d) / sizeof(double);
        __m512d sum = _mm512_setzero_pd();
        std::size_t i = 0;

        for (i = 0; i + VECTOR_SIZE <= LayerSize[layerIt - 1]; i += VECTOR_SIZE) {
            sum = _mm512_fmadd_pd(
                _mm512_loadu_pd(&neuronValue(layerIt - 1, i)), _mm512_loadu_pd(&neuronWeight(layerIt, neuronIt, i)), sum
            );
        }
        sigma += _mm512_reduce_add_pd(sum);
        for (; i < LayerSize[layerIt - 1]; ++i) {
            sigma += neuronValue(layerIt - 1, i) * neuronWeight(layerIt, neuronIt, i);
        }
        neuronValue(layerIt, neuronIt) = _sigmoid(sigma);
    #else
        double sigma = neuronBias(layerIt, neuronIt);

        for (std::size_t weightIt = 0; weightIt < LayerSize[layerIt - 1]; ++weightIt) {
            sigma += neuronValue(layerIt - 1, weightIt) * neuronWeight(layerIt, neuronIt, weightIt);
        }
        neuronValue(layerIt, neuronIt) = _sigmoid(sigma);
    #endif
    }

    public:
    NeuralNetwork(double learningRate, double minLearningRate, double maxLearningRate, double momentum = 0.9)
        : learningRate(learningRate), _minLearningRate(minLearningRate), _maxLearningRate(maxLearningRate),
          _momentum(momentum)
    {
        neuronBias.fillRandomNormalized();
        neuronMomentumBias.fill(0);
        neuronValue.fill(0);
        neuronDelta.fill(0);
        neuronWeight.fillRandomNormalized();
        neuronMomentumWeight.fill(0);
    }

    void updateLearningRate(double currentErrorRate)
    {
        std::uniform_real_distribution<double> dist(-0.1, 0.1);

        _temperature = _getTemperature();
        if (currentErrorRate < _prevErrorRate) {
            learningRate = std::min(_maxLearningRate, learningRate * 1.05);
        } else {
            learningRate = std::max(_minLearningRate, learningRate / 2.0);
            learningRate =
                std::clamp(learningRate * (1.0 + (dist(_gen) * _temperature)), _minLearningRate, _maxLearningRate);
        }
        _prevErrorRate = currentErrorRate;
        ++_epoch;
    }

    double computeMSE(const std::array<double, LayerSize.back()> &target) const
    {
        {
            double error = 0;
            double diff = 0;

            for (std::size_t i = 0; i < LayerSize.back(); ++i) {
                diff = neuronValue.back(i) - target[i];
                error += diff * diff;
            }
            return error / static_cast<double>(LayerSize.back());
        }
    }

    void computeOutput(const std::array<double, LayerSize.front()> &input)
    {
        _setInput(input);
        for (std::size_t layerIt = 1; layerIt < NbLayer; ++layerIt) {
            for (std::size_t neuronIt = 0; neuronIt < LayerSize[layerIt]; ++neuronIt) {
                _computeNeuronValue(layerIt, neuronIt);
            }
        }
    }

    void computeOutputDelta(const std::array<double, LayerSize.back()> &target)
    {
        for (std::size_t i = 0; i < LayerSize.back(); ++i) {
            neuronDelta.back(i) = (target[i] - neuronValue.back(i)) * _sigmoidDerivate(neuronValue.back(i));
        }
    }

    void computeDelta(std::size_t currentLayer, std::size_t nextLayer)
    {
        double error = 0;

        for (std::size_t i = 0; i < LayerSize[currentLayer]; ++i) {
            error = 0;
            for (std::size_t j = 0; j < LayerSize[nextLayer]; ++j) {
                error += neuronDelta(nextLayer, j) * neuronWeight(nextLayer, j, i);
            }
            neuronDelta(currentLayer, i) = error * _sigmoidDerivate(neuronValue(currentLayer, i));
        }
    }

    void updateParameter(std::size_t inputLayer, std::size_t currentLayer, std::size_t currentNeuron, double delta)
    {
        const double deltaMultlearningRate = learningRate * delta;

        for (std::size_t weightIt = 0; weightIt < LayerSize[currentLayer - 1]; ++weightIt) {
            neuronMomentumWeight(currentLayer, currentNeuron, weightIt) =
                (_momentum * neuronMomentumWeight(currentLayer, currentNeuron, weightIt)) +
                (deltaMultlearningRate * neuronValue(inputLayer, weightIt));
            neuronWeight(currentLayer, currentNeuron, weightIt) +=
                neuronMomentumWeight(currentLayer, currentNeuron, weightIt);
        }
        neuronMomentumBias(currentLayer, currentNeuron) =
            (_momentum * neuronMomentumBias(currentLayer, currentNeuron)) + (deltaMultlearningRate);
        neuronBias(currentLayer, currentNeuron) += neuronMomentumBias(currentLayer, currentNeuron);
    }

    void updateLayerParameter(std::size_t inputLayer, std::size_t currentLayer)
    {
        for (std::size_t i = 0; i < LayerSize[currentLayer]; ++i) {
            updateParameter(inputLayer, currentLayer, i, neuronDelta(currentLayer, i));
        }
    }

    void train(const std::array<double, LayerSize.front()> &input, const std::array<double, LayerSize.back()> &target)
    {
        computeOutput(input);

        computeOutputDelta(target);
        for (std::size_t i = NbLayer - 2; i > 0; --i) {
            computeDelta(i, i + 1);
        }
        for (std::size_t layerIt = 1; layerIt < NbLayer; ++layerIt) {
            updateLayerParameter(layerIt - 1, layerIt);
        }
    }

    long getIndexMaxOutputValue()
    {
        double maxValue = 0;
        std::size_t indexMaxValue = 0;

        for (std::size_t i = 0; i < LayerSize.back(); ++i) {
            if (neuronValue.back(i) > maxValue) {
                maxValue = neuronValue.back(i);
                indexMaxValue = i;
            }
        }
        return static_cast<long>(indexMaxValue);
    }

    LearningState computeLearningState(
        const std::vector<TrainingData<LayerSize.front(), LayerSize.back()>> &evaluateData
    )
    {
        LearningState learningState;

        for (const auto &it : evaluateData) {
            computeOutput(it.first);
            learningState.accuracyRate += computeMSE(it.second);
            if (getIndexMaxOutputValue() ==
                std::distance(it.second.begin(), std::max_element(it.second.begin(), it.second.end()))) {
                learningState.successRate += 1;
            }
        }
        learningState.accuracyRate /= static_cast<double>(evaluateData.size());
        learningState.successRate /= static_cast<double>(evaluateData.size());

        learningState.accuracyRate = (1 - learningState.accuracyRate);
        return learningState;
    }

    static void drawBar(double progressLevel, int barSize, double medium, double high)
    {
        int progress = 0;
        std::string bar;

        progress = static_cast<int>(std::round(progressLevel * barSize));
        if (progressLevel > high) {
            bar += "\e[92m";
        } else if (progressLevel > medium) {
            bar += "\e[93m";
        } else {
            bar += "\e[91m";
        }
        for (int i = 0; i < progress; ++i) {
            bar += "█";
        }
        bar += "\e[0m";
        for (int i = 0; i < barSize - progress; ++i) {
            bar += "▒";
        }
        std::cout << "[" << bar << "] " << std::setw(10) << std::setprecision(6) << std::fixed << progressLevel * 100.0
                  << "%";
    }

    static void drawProgressBar(double progressLevel, int barSize)
    {
        int progress = 0;
        std::string bar;

        progress = static_cast<int>(std::round(progressLevel * barSize));
        bar += "\e[2m";
        for (int i = 0; i < progress; ++i) {
            bar += "█";
        }
        for (int i = 0; i < barSize - progress; ++i) {
            bar += "▒";
        }
        std::cout << "[" << bar << "]\e[0m " << std::setw(6) << std::setprecision(2) << std::fixed
                  << progressLevel * 100.0 << "%";
    }

    void printProgress(
        double successRate,
        double accuracyRate,
        std::size_t trainingProgress,
        std::size_t batch,
        int barSize
    ) const
    {
        std::cout << "Progress:";
        drawProgressBar(static_cast<double>(trainingProgress) / static_cast<double>(batch), barSize);
        std::cout << ", Success Rate:";
        drawBar(successRate, barSize, 0.70, 0.90);
        std::cout << ", Accuracy Rate:";
        drawBar(accuracyRate, barSize, 0.70, 0.90);
        std::cout << ", learning Rate:" << learningRate << "\n";
    }

    void encode(const std::string &filename) const
    {
        std::ofstream file(filename);

        if (!file.is_open()) {
            throw utils::TrackedException("Failed to open file: " + filename + '.');
        }
        file << "learning_rate: " << learningRate << '\n';
        file << "min_learning_rate: " << _minLearningRate << '\n';
        file << "max_learning_rate: " << _maxLearningRate << '\n';
        file << "momentum: " << _momentum << '\n';
        file << "nb_layer: " << NbLayer << '\n';
        for (std::size_t i = 0; i < NbLayer; ++i) {
            file << "nb_neuron: " << LayerSize[i] << '\n';
            if (i == 0) {
                continue;
            }
            for (std::size_t j = 0; j < LayerSize[i]; ++j) {
                file << "    nb_weight: " << LayerSize[i];
                for (std::size_t weightIt = 0; weightIt < LayerSize[i]; ++weightIt) {
                    file << ", " << neuronWeight(i, j, weightIt);
                }
                file << '\n';
                file << "    nb_weight_momentum: " << LayerSize[i];
                for (std::size_t weightIt = 0; weightIt < LayerSize[i]; ++weightIt) {
                    file << ", " << neuronMomentumWeight(i, j, weightIt);
                }
                file << '\n';
                file << "    bias: " << neuronBias(i, j) << '\n';
                file << "    bias_momentum: " << neuronMomentumBias(i, j) << '\n';
            }
            file << '\n';
        }
    }
};

} // namespace nn
#endif
