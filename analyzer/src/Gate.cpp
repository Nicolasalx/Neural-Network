/*
** EPITECH PROJECT, 2024
** B-CNA-500-PAR-5-1-neuralnetwork-thibaud.cathala
** File description:
** Gate
*/

#include "Gate.hpp"
#include <utility>

std::vector<std::pair<std::vector<double>, std::vector<double>>> train::gate::getTrainigData()
{
    std::vector<std::pair<std::vector<double>, std::vector<double>>> trainingData;

    /*
        Input Output
        A B | A XOR B
        0 0 | 0
        0 1 | 1
        1 0 | 1
        1 1 | 0
    */
    trainingData.push_back({{0, 0}, {0}});
    trainingData.push_back({{0, 1}, {1}});
    trainingData.push_back({{1, 0}, {1}});
    trainingData.push_back({{1, 1}, {0}});
    return trainingData;
}
