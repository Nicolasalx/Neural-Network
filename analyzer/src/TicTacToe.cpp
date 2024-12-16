/*
** EPITECH PROJECT, 2024
** B-CNA-500-PAR-5-1-neuralnetwork-thibaud.cathala
** File description:
** TicTacToe
*/

#include "TicTacToe.hpp"
#include <array>
#include <cstddef>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <vector>
#ifdef STATIC_NEURAL_NETWORK
    #include "StaticNeuralNetwork.hpp"
#else
    #include "NeuralNetwork.hpp"
#endif
#include "TrackedException.hpp"

static nn::TrainingData<train::tictactoe::NB_INPUT, train::tictactoe::NB_OUTPUT> parseLine(const std::string &line)
{
    nn::TrainingData<train::tictactoe::NB_INPUT, train::tictactoe::NB_OUTPUT> data;
    /*
        IN 1-4-7-...-25: IS_EMPTY
        IN 2-5-8-...-26: X
        IN 3-6-9-...-27: O
    */
    /*
        OUT 1: Draw
        OUT 2: Win X
        OUT 3: Win O
    */
    data.first.fill(0);
    data.second.fill(0);
    for (std::size_t i = 0; i < 9; ++i) {
        switch (line[i]) {
            case 'X':
                data.first[(i * 3) + 1] = 1;
                break;
            case 'O':
                data.first[(i * 3) + 2] = 1;
                break;
            default:
                data.first[i * 3] = 1;
                break;
        }
    }
    if (line[9] == 'D') {
        data.second[0] = 1;
    } else if (line[9] == 'W') {
        if (line[10] == 'X') {
            data.second[1] = 1;
        } else {
            data.second[2] = 1;
        }
    }
    return data;
}

std::vector<nn::TrainingData<train::tictactoe::NB_INPUT, train::tictactoe::NB_OUTPUT>> train::tictactoe::getTrainigData(
    const std::string &filename
)
{
    std::vector<nn::TrainingData<NB_INPUT, NB_OUTPUT>> trainingData;
    std::ifstream file(filename);
    std::string line;

    if (!file.is_open()) {
        throw utils::TrackedException("Fail to open file: " + filename + ".");
    }
    while (std::getline(file, line)) {
        trainingData.push_back(parseLine(line));
    }
    return trainingData;
}

#ifdef STATIC_NEURAL_NETWORK
void train::tictactoe::displayResult(
    const std::string &filename,
    nn::NeuralNetwork<NB_LAYER, NN_LAYOUT> &neuralNetwork
)
#else
void train::tictactoe::displayResult(const std::string &filename, nn::NeuralNetwork &neuralNetwork)
#endif
{
    /*
        IN 1-4-7-...-25: IS_EMPTY
        IN 2-5-8-...-26: X
        IN 3-6-9-...-27: O
    */
    /*
        OUT 1: Draw
        OUT 2: Win X
        OUT 3: Win O
    */
    nn::TrainingData<NB_INPUT, NB_OUTPUT> trainingData;
    std::ifstream file(filename);
    std::string line;

    if (!file.is_open()) {
        throw utils::TrackedException("Fail to open file: " + filename + ".");
    }
    for (std::size_t i = 0; std::getline(file, line) && i < 100; ++i) {
        trainingData = parseLine(line);
        neuralNetwork.computeOutput(trainingData.first);
        std::cout << std::setw(11) << line << ", out: ";
#ifdef STATIC_NEURAL_NETWORK
        for (std::size_t itValue = 0; itValue < NB_OUTPUT; ++itValue) {
            std::cout << neuralNetwork.neuronValue.back(itValue) << ", ";
        }
#else
        for (const auto &it : neuralNetwork.neuronValue.back().content) {
            std::cout << it << ", ";
        }
#endif
        std::cout << "ref: ";
        for (const auto &it : trainingData.second) {
            std::cout << it << ", ";
        }
        if (
#ifdef STATIC_NEURAL_NETWORK
            neuralNetwork.getIndexMaxOutputValue() ==
#else
            std::distance(
                neuralNetwork.neuronValue.back().content.begin(),
                std::max_element(
                    neuralNetwork.neuronValue.back().content.begin(), neuralNetwork.neuronValue.back().content.end()
                )
            ) ==
#endif
            std::distance(
                trainingData.second.begin(), std::max_element(trainingData.second.begin(), trainingData.second.end())
            )) {
            std::cout << " \e[92m█\e[0m";
        } else {
            std::cout << " \e[91m█\e[0m";
        }
        std::cout << '\n';
    }
}
