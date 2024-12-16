/*
** EPITECH PROJECT, 2024
** B-CNA-500-PAR-5-1-neuralnetwork-thibaud.cathala
** File description:
** main
*/

#include <cmath>
#include <string>
#include "ArgParsing.hpp"
#include "Chess.hpp"
#include "Logger.hpp"

// constexpr std::size_t BATCH = 10000;

// void trainLoop(nn::NeuralNetwork &neuralNetwork, const char **argv, train::Metric &metric, std::size_t trainingI)
// {
//     constexpr std::size_t CHUNCK_SIZE = 10'000;
//     train::Chess chessTraining(argv[1], CHUNCK_SIZE);
//     train::Chess chessEvaluate(argv[2], CHUNCK_SIZE);
//     nn::NeuralNetwork::LearningState learningState;
//     double mse = 0;
//     double nbMse = 0;
//     double progress = CHUNCK_SIZE;
//     double maxProgresse = 507'702;

//     chessTraining.getTrainigData(
//         [&neuralNetwork, &mse, &nbMse, &progress, &maxProgresse](
//             const std::vector<nn::TrainingData<train::Chess::NB_INPUT, train::Chess::NB_OUTPUT>> &trainingData
//         ) {
//             for (const auto &it : trainingData) {
//                 neuralNetwork.train(it.first, it.second);
//                 mse += neuralNetwork.computeMSE(it.second);
//                 ++nbMse;
//             }
//             std::cout << (progress / maxProgresse) * 100.0 << " %" << std::endl;
//             progress += CHUNCK_SIZE;
//         }
//     );
//     mse /= nbMse;

//     // std::cout << "Start Evaluating" << std::endl;
//     chessEvaluate.getTrainigData(
//         [&neuralNetwork, &learningState](
//             const std::vector<nn::TrainingData<train::Chess::NB_INPUT, train::Chess::NB_OUTPUT>> &evaluateData
//         ) {
//             learningState += neuralNetwork.computeLearningState(evaluateData);
//             // successRate += neuralNetwork.computeSuccessRate(evaluateData);
//         }
//     );
//     learningState /= chessEvaluate.getNbChunkRetreived();
//     // successRate /= chessEvaluate.getNbChunkRetreived();

//     neuralNetwork.updateLearningRate(1 - learningState.accuracyRate);

//     // !
//     metric.setMetric(trainingI, mse, learningState.successRate, learningState.accuracyRate, neuralNetwork.learningRate);
//     neuralNetwork.printProgress(learningState.successRate, learningState.accuracyRate, trainingI, BATCH, 20);
//     neuralNetwork.encode(
//         "chess_lastest_" + std::to_string(std::hash<std::thread::id>{}(std::this_thread::get_id())) + ".nn"
//     );
// }

// int trainingAndEvaluation(int argc, const char **argv)
// {
//     nn::NeuralNetwork neuralNetwork(
//         {train::Chess::NB_INPUT, 578, 256, 64, train::Chess::NB_OUTPUT}, 0.1, 0.000'001, 10, 1
//     );
//     train::Metric metric(
//         "chess_lastest_" + std::to_string(std::hash<std::thread::id>{}(std::this_thread::get_id())) + ".csv"
//     );

//     if (argc != 3) {
//         return 84;
//     }
//     for (std::size_t trainingI = 1; trainingI <= BATCH; ++trainingI) {
//         trainLoop(neuralNetwork, argv, metric, trainingI);
//     }
//     train::Chess::displayResult(argv[2], neuralNetwork);
//     return 0;
// }

// ./my_torch_analyzer [--predict | --train [--save SAVEFILE]] LOADFILE FILE
// ./my_torch_analyzer --predict LOADFILE FILE
// ./my_torch_analyzer --predict --color LOADFILE FILE
// ./my_torch_analyzer --train LOADFILE FILE
// ./my_torch_analyzer --train [--save SAVEFILE] LOADFILE FILE
// ./my_torch_analyzer --train_stat SAVEFILE TRAINING_FILE EVALUATE_FILE

int main(int argc, const char *argv[])
{
    ArgParsing arg;

    arg.registerOption("--predict", train::Chess::predictBoardState);
    arg.registerOption("--train", train::Chess::trainNeuralNetwork);
    arg.registerOption("--train_stat", train::Chess::trainStatNeuralNetwork);
    try {
        arg.handleArg(argc, argv);
    } catch (const std::exception &e) {
        arg.printHelp();
        utils::logError(e.what());
        return 84;
    }
    return 0;
}
