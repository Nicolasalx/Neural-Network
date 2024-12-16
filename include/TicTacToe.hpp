/*
** EPITECH PROJECT, 2024
** B-CNA-500-PAR-5-1-neuralnetwork-thibaud.cathala
** File description:
** TicTacToe
*/

#pragma once

#include <string>
#include <vector>
#ifdef STATIC_NEURAL_NETWORK
#include "StaticNeuralNetwork.hpp"
#else
#include "NeuralNetwork.hpp"
#endif

namespace train::tictactoe {

constexpr std::size_t NB_INPUT = 27;
constexpr std::size_t NB_OUTPUT = 3;

static constexpr std::size_t NB_LAYER = 3;
static constexpr std::array<std::size_t, NB_LAYER> NN_LAYOUT = {NB_INPUT, 32, NB_OUTPUT};

std::vector<nn::TrainingData<train::tictactoe::NB_INPUT, train::tictactoe::NB_OUTPUT>> getTrainigData(
    const std::string &filename
);
#ifdef STATIC_NEURAL_NETWORK
void displayResult(const std::string &filename, nn::NeuralNetwork<NB_LAYER, NN_LAYOUT> &neuralNetwork);
#else
void displayResult(const std::string &filename, nn::NeuralNetwork &neuralNetwork);
#endif

} // namespace train::tictactoe
