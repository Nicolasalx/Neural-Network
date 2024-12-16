/*
** EPITECH PROJECT, 2024
** B-CNA-500-PAR-5-1-neuralnetwork-thibaud.cathala
** File description:
** train_neural_network
*/

#include <iostream>
#include <string>
#include <thread>
#include <utility>
#include <vector>
#include "Chess.hpp"
#include "LTDManager.hpp"
#include "Metric.hpp"
#include "TrainingData.hpp"

void trainLoop(
    nn::NeuralNetwork &neuralNetwork,
    const std::vector<nn::TrainingData<train::Chess::NB_INPUT, train::Chess::NB_OUTPUT>> &trainingData,
    const std::vector<nn::TrainingData<train::Chess::NB_INPUT, train::Chess::NB_OUTPUT>> &evaluateData,
    const std::pair<std::size_t, std::size_t> trainingItEpoch,
    const std::string &saveFile,
    train::Metric &metric
)
{
    nn::NeuralNetwork::LearningState learningState;
    double mse = 0;

    // std::cout << "\033[s";
    for (std::size_t i = 0; i < trainingData.size(); ++i) {
        neuralNetwork.train(trainingData[i].first, trainingData[i].second);
        mse += neuralNetwork.computeMSE(trainingData[i].second);
        // if (i % 100 == 0) {
        //     std::cout << "\033[u" << std::setw(10) << std::setprecision(2) << std::fixed
        //               << (static_cast<double>(i) / static_cast<double>(trainingData.size())) * 100 << " %";
        // }
    }
    // std::cout << "\033[u";
    mse /= static_cast<double>(trainingData.size());

    learningState = neuralNetwork.computeLearningState(evaluateData);

    neuralNetwork.updateLearningRate(1 - learningState.accuracyRate);

    metric.setMetric(
        trainingItEpoch.first, mse, learningState.successRate, learningState.accuracyRate, neuralNetwork.learningRate
    );
    neuralNetwork.printProgress(
        learningState.successRate, learningState.accuracyRate, trainingItEpoch.first, trainingItEpoch.second, 20
    );
    neuralNetwork.encode(saveFile);
}

// ./my_torch_analyzer --train_stat [SAVEFILE, ...] TRAINING_FILE EVALUATE_FILE
void train::Chess::trainStatNeuralNetwork(int argc, const char **argv)
{
    constexpr std::size_t NB_EPOCH = 1000;
    int nnFileIndex = 2;
    int trainingFileIndex = argc - 2;
    int evaluateFileIndex = argc - 1;
    std::vector<std::thread> thread;

    if (argc < 5) {
        throw utils::TrackedException("Invalid number of arguments for --train_stat.");
    }
    LTDManager ltdManager(argv[trainingFileIndex], 10'000);
    std::vector<nn::TrainingData<NB_INPUT, NB_OUTPUT>> trainingData =
        ltdManager.getEntireTrainingData<NB_INPUT, NB_OUTPUT>(parseFen);
    std::cout << "Training data size: "
              << static_cast<double>(trainingData.size() * sizeof(nn::TrainingData<NB_INPUT, NB_OUTPUT>)) / 1'000'000.0
              << " MB" << std::endl;
    std::cout << "Training data memory comsumption: "
              << static_cast<double>(trainingData.capacity() * sizeof(nn::TrainingData<NB_INPUT, NB_OUTPUT>)) /
            1'000'000.0
              << " MB" << std::endl;
    LTDManager ltdManagerEvaluate(argv[evaluateFileIndex], 10'000);
    std::vector<nn::TrainingData<NB_INPUT, NB_OUTPUT>> evaluateData =
        ltdManagerEvaluate.getEntireTrainingData<NB_INPUT, NB_OUTPUT>(parseFen);
    std::cout << "Evaluation data size: "
              << static_cast<double>(evaluateData.size() * sizeof(nn::TrainingData<NB_INPUT, NB_OUTPUT>)) / 1'000'000.0
              << " MB" << std::endl;
    std::cout << "Evaluation data memory comsumption: "
              << static_cast<double>(evaluateData.capacity() * sizeof(nn::TrainingData<NB_INPUT, NB_OUTPUT>)) /
            1'000'000.0
              << " MB" << std::endl;
    for (int nnIt = nnFileIndex; nnIt < argc - 2; ++nnIt) {
        std::cout << "Launching training of " << argv[nnIt] << std::endl;
        thread.emplace_back([nnIt, &argv, NB_EPOCH, &trainingData, &evaluateData] {
            nn::NeuralNetwork neuralNetwork(argv[nnIt], NB_INPUT, NB_OUTPUT);
            train::Metric metric(std::string(argv[nnIt]).substr(0, std::string(argv[nnIt]).size() - 3) + ".csv");

            for (std::size_t trainingIt = 1; trainingIt <= NB_EPOCH; ++trainingIt) {
                trainLoop(neuralNetwork, trainingData, evaluateData, {trainingIt, NB_EPOCH}, argv[nnIt], metric);
            }
        });
    }
    for (auto &it : thread) {
        if (it.joinable()) {
            it.join();
        }
    }
}
