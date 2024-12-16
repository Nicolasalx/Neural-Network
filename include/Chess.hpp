/*
** EPITECH PROJECT, 2024
** B-CNA-500-PAR-5-1-neuralnetwork-thibaud.cathala
** File description:
** Chess
*/

#pragma once

#include <array>
#include <cstddef>
#include <map>
#include <string>
#include "NeuralNetwork.hpp"
#include <string_view>

namespace train {

class Chess {
    private:
    static constexpr std::size_t BOARD_SIZE = 8.0 * 8.0;
    static constexpr std::size_t STATE_PER_CELL = 7;

    public:
    static constexpr std::size_t NB_INPUT = (BOARD_SIZE * STATE_PER_CELL) + 2;
    static constexpr std::size_t NB_OUTPUT = 6;

    static constexpr std::size_t NB_LAYER = 5;

    using ChessTrainingData = nn::TrainingData<Chess::NB_INPUT, Chess::NB_OUTPUT>;

    private:
    static constexpr std::size_t NEXT_PLAY_W = NB_INPUT - 2;
    static constexpr std::size_t NEXT_PLAY_B = NB_INPUT - 1;

    static constexpr std::array<std::string_view, 4> OUTPUT_STATE = {"Checkmate", "Check", "Stalemate", "Nothing"};
    static constexpr std::array<std::string_view, 2> OUTPUT_COLOR_STATE = {"White", "Black"};
    static constexpr std::array<std::string_view, 6> OUTPUT_PREDICTION =
        {"Checkmate White", "Checkmate Black", "Check White", "Check Black", "Stalemate", "Nothing"};

    static const std::map<char, std::array<bool, train::Chess::STATE_PER_CELL>> CELL_STATE;

    static void displayPrediction(
        nn::NeuralNetwork &neuralNetwork,
        const std::vector<nn::TrainingData<NB_INPUT, NB_OUTPUT>> &evaluateData
    );
    static void predictBoardStateColor(
        nn::NeuralNetwork &neuralNetwork,
        const std::vector<nn::TrainingData<NB_INPUT, NB_OUTPUT>> &evaluateData
    );

    public:
    Chess() = default;

    static nn::TrainingData<NB_INPUT, NB_OUTPUT> parseFen(const std::string &str);

    static void trainNeuralNetwork(int argc, const char **argv);
    static void trainStatNeuralNetwork(int argc, const char **argv);
    static void predictBoardState(int argc, const char **argv);

    static void displayResult(const std::string &filename, nn::NeuralNetwork &neuralNetwork);
};

} // namespace train
