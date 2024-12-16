/*
** EPITECH PROJECT, 2024
** B-CNA-500-PAR-5-1-neuralnetwork-thibaud.cathala
** File description:
** train_neural_network
*/

#include <string>
#include <vector>
#include "Chess.hpp"
#include "LTDManager.hpp"

void trainLoop(
    nn::NeuralNetwork &neuralNetwork,
    const std::vector<nn::TrainingData<train::Chess::NB_INPUT, train::Chess::NB_OUTPUT>> &trainingData,
    const std::size_t trainingIt,
    const std::size_t nbEpoch,
    const std::string &saveFile
)
{
    // nn::NeuralNetwork::LearningState learningState;
    double mse = 0;

    for (const auto &it : trainingData) {
        neuralNetwork.train(it.first, it.second);
        mse += neuralNetwork.computeMSE(it.second);
    }
    mse /= static_cast<double>(trainingData.size());

    // learningState = neuralNetwork.computeLearningState(evaluateData);

    neuralNetwork.updateLearningRate(1 - mse/* learningState.accuracyRate */);

    neuralNetwork.printProgress(/* learningState.successRate, learningState.accuracyRate */ mse, mse, trainingIt, nbEpoch, 20);
    neuralNetwork.encode(saveFile);
}

// ./my_torch_analyzer --train LOADFILE FILE
// ./my_torch_analyzer --train [--save SAVEFILE] LOADFILE FILE
void train::Chess::trainNeuralNetwork(int argc, const char **argv)
{
    constexpr std::size_t NB_EPOCH = 1000;
    int saveFileIndex = 2;
    int nnFileIndex = 2;
    int trainingFileIndex = 3;

    if (argc == 4) {
    } else if (argc == 6) {
        saveFileIndex = 3;
        nnFileIndex = 4;
        trainingFileIndex = 5;
    } else {
        throw utils::TrackedException("Invalid number of arguments.");
    }
    nn::NeuralNetwork neuralNetwork(argv[nnFileIndex], NB_INPUT, NB_OUTPUT);
    LTDManager ltdManager(argv[trainingFileIndex], 10'000);
    std::vector<nn::TrainingData<NB_INPUT, NB_OUTPUT>> trainingData =
        ltdManager.getEntireTrainingData<NB_INPUT, NB_OUTPUT>(parseFen);

    for (std::size_t trainingIt = 1; trainingIt <= NB_EPOCH; ++trainingIt) {
        trainLoop(neuralNetwork, trainingData, trainingIt, NB_EPOCH, argv[saveFileIndex]);
    }
}
