/*
** EPITECH PROJECT, 2024
** B-CNA-500-PAR-5-1-neuralnetwork-thibaud.cathala
** File description:
** chess
*/

#include "Chess.hpp"
#include <array>
#include <cstddef>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>
#include "NeuralNetwork.hpp"
#include "SplitString.hpp"
#include "TrackedException.hpp"

/*
    CELL: 0,0
    0: empty
    1: White
    3: Black
    4: P
    5: N
    6: B
    7: R
    8: Q
    9: K
*/

const std::map<char, std::array<bool, train::Chess::STATE_PER_CELL>> train::Chess::CELL_STATE = {
    {'P', {true, false, false, false, false, false, true}},
    {'N', {false, true, false, false, false, false, true}},
    {'B', {false, false, true, false, false, false, true}},
    {'R', {false, false, false, true, false, false, true}},
    {'Q', {false, false, false, false, true, false, true}},
    {'K', {false, false, false, false, false, true, true}},
    {'p', {true, false, false, false, false, false, false}},
    {'n', {false, true, false, false, false, false, false}},
    {'b', {false, false, true, false, false, false, false}},
    {'r', {false, false, false, true, false, false, false}},
    {'q', {false, false, false, false, true, false, false}},
    {'k', {false, false, false, false, false, true, false}} 
};

nn::TrainingData<train::Chess::NB_INPUT, train::Chess::NB_OUTPUT> train::Chess::parseFen(const std::string &str)
{
    nn::TrainingData<NB_INPUT, NB_OUTPUT> trainingData;
    std::vector<std::string> lineContent = utils::splitString(str, " ");
    constexpr std::size_t CELL_STATE_IT = 0;
    constexpr std::size_t NEXT_PLAY_IT = 1;
    constexpr std::size_t END_STATE_IT = 6;
    constexpr std::size_t END_STATE_COLOR_IT = 7;
    int cellIndex = 0;

    // ! 3q4/3P1p2/p1PQ4/5kp1/6Pp/8/6PK/8 b - - 0 39 Check White
    // ! ^0                               ^1^2^3^4^5 ^6    ^7
    // ! ^1                               ^2^3^4^5^6 ^7    ^8

    trainingData.first.fill(0);
    trainingData.second.fill(0);
    if (lineContent.size() < 6 || lineContent.size() > 8) {
        throw utils::TrackedException("Invalid FEN format.");
    }
    for (char c : lineContent[CELL_STATE_IT]) {
        if (c == '/') {
            continue;
        } else if (std::isdigit(c)) {
            cellIndex += (c - '0');
        } else if (CELL_STATE.contains(c)) {
            const auto &encoding = CELL_STATE.at(c);
            for (std::size_t feature = 0; feature < STATE_PER_CELL; ++feature) {
                trainingData.first[cellIndex * STATE_PER_CELL + feature] = encoding[feature];
            }
            ++cellIndex;
        }
    }

    if (lineContent[NEXT_PLAY_IT] == "w") {
        trainingData.first[NEXT_PLAY_W] = true;
    } else {
        trainingData.first[NEXT_PLAY_B] = true;
    }

    if (lineContent.size() == 6) {
        return trainingData;
    }
    if (lineContent[END_STATE_IT] == OUTPUT_STATE[0]) {                 // Checkmate
        if (lineContent[END_STATE_COLOR_IT] == OUTPUT_COLOR_STATE[1]) { // Black
            trainingData.second[1] = true;
        } else {
            trainingData.second[0] = true; // White
        }
    } else if (lineContent[END_STATE_IT] == OUTPUT_STATE[1]) {          // Check
        if (lineContent[END_STATE_COLOR_IT] == OUTPUT_COLOR_STATE[1]) { // Black
            trainingData.second[3] = true;
        } else {
            trainingData.second[2] = true; // White
        }
    } else if (lineContent[END_STATE_IT] == OUTPUT_STATE[2]) { // Stalemate
        trainingData.second[4] = true;
    } else if (lineContent[END_STATE_IT] == OUTPUT_STATE[3]) {
        trainingData.second[5] = true;
    }
    // if (END_STATE == OUTPUT_STATE[0] || END_STATE == OUTPUT_STATE[1]) {
    //     trainingData.second[0] = 1;
    // }
    // if (END_STATE == OUTPUT_STATE[2] || END_STATE == OUTPUT_STATE[3]) {
    //     trainingData.second[1] = 1;
    // }
    return trainingData;
}

void train::Chess::displayResult(const std::string &filename, nn::NeuralNetwork &neuralNetwork)
{
    /*
        OUT 1: Checkmate
        OUT 2: Check
        OUT 3: Stalemate
        OUT 4: Nothing
    */
    nn::TrainingData<NB_INPUT, NB_OUTPUT> trainingData;
    std::ifstream file(filename);
    std::string line;

    if (!file.is_open()) {
        throw utils::TrackedException("Fail to open file: " + filename + ".");
    }
    for (std::size_t i = 0; std::getline(file, line) && i < 100; ++i) {
        trainingData = parseFen(line);
        neuralNetwork.computeOutput(trainingData.first);
        std::cout << std::setw(80) << line << ", out: ";
        for (const auto &it : neuralNetwork.neuronValue.back().content) {
            std::cout << it << ", ";
        }
        std::cout << "ref: ";
        for (const auto &it : trainingData.second) {
            std::cout << it << ", ";
        }
        if (neuralNetwork.getMaxOutputIndex() ==
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
