/*
** EPITECH PROJECT, 2024
** B-CNA-500-PAR-5-1-neuralnetwork-thibaud.cathala
** File description:
** create_from_conf
*/

#include <fstream>
#include "NeuralNetwork.hpp"
#include "SplitString.hpp"
#include "TrackedException.hpp"

void nn::NeuralNetwork::_createFromConf(const std::string &filename)
{
    std::ifstream file(filename);
    std::vector<std::string> word;
    std::string line;
    std::vector<std::size_t> layerSize;

    if (!file.is_open()) {
        throw utils::TrackedException("Fail to open file: " + filename + ".");
    }
    if (!std::getline(file, line)) {
        throw utils::TrackedException("Invalid file content: " + filename + ".");
    }
    word = utils::splitString(line, ",;:| \n\t");
    if (word.size() != 4) {
        throw utils::TrackedException("Invalid Hyper Parameter: \"" + line + "\".");
    }
    learningRate = std::stod(word[0]);
    _minLearningRate = std::stod(word[1]);
    _maxLearningRate = std::stod(word[2]);
    _momentum = std::stod(word[3]);
    if (!std::getline(file, line)) {
        throw utils::TrackedException("Invalid Layer: \"" + filename + "\".");
    }
    word = utils::splitString(line, ",;:| \n\t");
    for (const auto &it : word) {
        if (std::stol(it) < 1) {
            throw utils::TrackedException("Invalid Layer Size: \"" + it + "\".");
        }
        layerSize.push_back(std::stoul(it));
    }
    _createNeuron(layerSize);
}
