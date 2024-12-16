/*
** EPITECH PROJECT, 2024
** B-CNA-500-PAR-5-1-neuralnetwork-thibaud.cathala
** File description:
** predict_chess_board
*/

#include <array>
#include <iomanip>
#include <iostream>
#include "Chess.hpp"
#include "LTDManager.hpp"
#include "NeuralNetwork.hpp"

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
    std::cout << "[" << bar << "] " << std::setw(6) << std::setprecision(2) << std::fixed << progressLevel * 100.0
              << "%";
}

static bool validateHardState(std::size_t out, std::size_t ref)
{
    return (out == ref);
}

static bool validateMediumState(std::size_t out, std::size_t ref)
{
    return ((out == 0 || out == 1) && (ref == 0 || ref == 1)) || ((out == 2 || out == 3) && (ref == 2 || ref == 3)) ||
        validateHardState(out, ref);
}

static bool validateSimpleState(std::size_t out, std::size_t ref)
{
    return ((out == 2 || out == 3) && ref <= 3) || (out == 5 && ref >= 4 && ref <= 5) ||
        validateMediumState(out, ref) || validateHardState(out, ref);
}

void train::Chess::predictBoardStateColor(
    nn::NeuralNetwork &neuralNetwork,
    const std::vector<nn::TrainingData<NB_INPUT, NB_OUTPUT>> &evaluateData
)
{
    std::size_t out = 0;
    std::size_t ref = 0;
    std::array<int, 3> succesOut = {0};

    for (const auto &it : evaluateData) {
        neuralNetwork.computeOutput(it.first);
        out = neuralNetwork.getMaxOutputIndex();
        ref = std::distance(it.second.begin(), std::max_element(it.second.begin(), it.second.end()));
        std::cout << "ref: " << std::setw(15) << OUTPUT_PREDICTION[ref] << ", out: " << std::setw(15)
                  << OUTPUT_PREDICTION[out] << " |";
        for (std::size_t i = 0; i < NB_OUTPUT; ++i) {
            std::cout << " " << std::setw(6) << std::setprecision(2) << std::fixed
                      << neuralNetwork.neuronValue.back().content[i];
        }
        for (std::size_t i = 0; i < 3; ++i) {
            if ((i == 0 && validateSimpleState(out, ref)) || (i == 1 && validateMediumState(out, ref)) ||
                (i == 2 && validateHardState(out, ref))) {
                ++succesOut[i];
                std::cout << " \e[92m█\e[0m";
            } else {
                std::cout << " \e[91m█\e[0m";
            }
        }
        std::cout << '\n';
    }
    std::cout << "\nBasic:  ";
    drawBar(succesOut[0] / static_cast<double>(evaluateData.size()), 50, 0.7, 0.9);
    std::cout << " (" << succesOut[0] << " / " << evaluateData.size() << ")\nMedium: ";
    drawBar(succesOut[1] / static_cast<double>(evaluateData.size()), 50, 0.7, 0.9);
    std::cout << " (" << succesOut[1] << " / " << evaluateData.size() << ")\nHard:   ";
    drawBar(succesOut[2] / static_cast<double>(evaluateData.size()), 50, 0.7, 0.9);
    std::cout << " (" << succesOut[2] << " / " << evaluateData.size() << ")\n";
}

void train::Chess::displayPrediction(
    nn::NeuralNetwork &neuralNetwork,
    const std::vector<nn::TrainingData<NB_INPUT, NB_OUTPUT>> &evaluateData
)
{
    for (const auto &it : evaluateData) {
        neuralNetwork.computeOutput(it.first);
        std::cout << OUTPUT_PREDICTION[neuralNetwork.getMaxOutputIndex()] << '\n';
    }
}

void train::Chess::predictBoardState(int argc, const char **argv)
{
    if (argc != 4 && (argc != 5 && std::string(argv[2]) != "--color")) {
        throw utils::TrackedException("Invalid number of arguments.");
    }
    nn::NeuralNetwork neuralNetwork(argv[2 + (argc - 4)], NB_INPUT, NB_OUTPUT);
    LTDManager ltdManager(argv[3 + (argc - 4)], 10'000);
    std::vector<nn::TrainingData<NB_INPUT, NB_OUTPUT>> evaluateData =
        ltdManager.getEntireTrainingData<NB_INPUT, NB_OUTPUT>(parseFen);

    if (argc == 4) {
        displayPrediction(neuralNetwork, evaluateData);
    } else {
        predictBoardStateColor(neuralNetwork, evaluateData);
    }
}
