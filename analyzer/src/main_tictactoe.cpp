// /*
// ** EPITECH PROJECT, 2024
// ** B-CNA-500-PAR-5-1-neuralnetwork-thibaud.cathala
// ** File description:
// ** main
// */

// #include <cmath>
// #include <cstddef>
// #include <vector>
// #include "Metric.hpp"
// #ifdef STATIC_NEURAL_NETWORK
//     #include "StaticNeuralNetwork.hpp"
// #else
//     #include "NeuralNetwork.hpp"
// #endif
// #include "TicTacToe.hpp"

// constexpr std::size_t BATCH = 10;

// void trainLoop(
// #ifdef STATIC_NEURAL_NETWORK
//     nn::NeuralNetwork<train::tictactoe::NB_LAYER, train::tictactoe::NN_LAYOUT> &neuralNetwork,
// #else
//     nn::NeuralNetwork &neuralNetwork,
// #endif
//     train::Metric &metric,
//     std::size_t trainingI,
//     const std::vector<nn::TrainingData<train::tictactoe::NB_INPUT, train::tictactoe::NB_OUTPUT>> &trainingData,
//     const std::vector<nn::TrainingData<train::tictactoe::NB_INPUT, train::tictactoe::NB_OUTPUT>> &evaluateData
// )
// {
// #ifdef STATIC_NEURAL_NETWORK
//     nn::NeuralNetwork<train::tictactoe::NB_LAYER, train::tictactoe::NN_LAYOUT>::LearningState learningState;
// #else
//     nn::NeuralNetwork::LearningState learningState;
// #endif
//     double mse = 0;

//     for (const auto &it : trainingData) {
//         neuralNetwork.train(it.first, it.second);
//         mse += neuralNetwork.computeMSE(it.second);
//     }
//     mse /= static_cast<double>(trainingData.size());

//     learningState = neuralNetwork.computeLearningState(evaluateData);

//     neuralNetwork.updateLearningRate(1 - learningState.accuracyRate);

//     // !
//     metric.setMetric(trainingI, mse, learningState.successRate, learningState.accuracyRate, neuralNetwork.learningRate);
//     neuralNetwork.printProgress(learningState.successRate, learningState.accuracyRate, trainingI, BATCH, 20);
// }

// int main(int argc, const char *argv[])
// {
//     if (argc != 3) {
//         return 84;
//     }

// #ifdef STATIC_NEURAL_NETWORK
//     nn::NeuralNetwork<train::tictactoe::NB_LAYER, train::tictactoe::NN_LAYOUT> neuralNetwork(0.01, 0.000'000'1, 10);
// #else
//     nn::NeuralNetwork neuralNetwork(
//         {train::tictactoe::NB_INPUT, 32, train::tictactoe::NB_OUTPUT}, 0.01, 0.000'000'1, 10
//     );
// #endif
//     train::Metric metric("tictactoe_data.csv");
//     std::vector<nn::TrainingData<train::tictactoe::NB_INPUT, train::tictactoe::NB_OUTPUT>> trainingData =
//         train::tictactoe::getTrainigData(argv[1]);
//     std::vector<nn::TrainingData<train::tictactoe::NB_INPUT, train::tictactoe::NB_OUTPUT>> evaluateData =
//         train::tictactoe::getTrainigData(argv[2]);

//     for (std::size_t trainingI = 1; trainingI <= BATCH; ++trainingI) {
//         trainLoop(neuralNetwork, metric, trainingI, trainingData, evaluateData);
//     }
//     // neuralNetwork.encode("tictactoe_end.nn");
//     train::tictactoe::displayResult(argv[2], neuralNetwork);
//     return 0;
// }
