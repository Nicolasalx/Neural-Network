/*
** EPITECH PROJECT, 2024
** B-CNA-500-PAR-5-1-neuralnetwork-thibaud.cathala
** File description:
** create_from_nn
*/

#include <cstddef>
#include <fstream>
#include <string>
#include <utility>
#include <vector>
#include "NeuralNetwork.hpp"
#include "SplitString.hpp"
#include "TrackedException.hpp"

void nn::NeuralNetwork::_parseNeuronValueFromFile(
    const std::vector<std::string> &weight,
    const std::vector<std::string> &weightMomentum,
    const std::vector<std::string> &bias,
    const std::vector<std::string> &biasMomentum,
    const std::vector<std::size_t> &layerSize,
    const std::pair<std::size_t, std::size_t> &layerNeuronIt
)
{
    for (std::size_t weightIt = 0; weightIt < layerSize.at(layerNeuronIt.first - 1); ++weightIt) {
        neuronWeight[layerNeuronIt.first](layerNeuronIt.second, weightIt) = std::stod(weight.at(weightIt + 2));
        neuronMomentumWeight[layerNeuronIt.first](layerNeuronIt.second, weightIt) =
            std::stod(weightMomentum.at(weightIt + 2));
    }
    neuronBias[layerNeuronIt.first].content[layerNeuronIt.second] = std::stod(bias.at(1));
    neuronMomentumBias[layerNeuronIt.first].content[layerNeuronIt.second] = std::stod(biasMomentum.at(1));
}

void nn::NeuralNetwork::_createFromNN(const std::string &filename, int nbInput, int nbOutput)
{
    std::ifstream file(filename);
    std::string line;
    std::vector<std::string> word;
    std::vector<std::string> fileContent;
    int nbLayer = 0;
    std::vector<std::size_t> layerSize;
    int currentLine = 6;
    std::vector<std::pair<int, int>> neuronLine;

    if (!file.is_open()) {
        throw utils::TrackedException("Fail to open file: " + filename + ".");
    }
    while (std::getline(file, line)) {
        fileContent.push_back(line);
    }
    try {
        learningRate = std::stod(utils::splitString(fileContent.at(0), ",;:| \n\t").at(1));
        _minLearningRate = std::stod(utils::splitString(fileContent.at(1), ",;:| \n\t").at(1));
        _maxLearningRate = std::stod(utils::splitString(fileContent.at(2), ",;:| \n\t").at(1));
        _momentum = std::stod(utils::splitString(fileContent.at(3), ",;:| \n\t").at(1));
        nbLayer = std::stoi(utils::splitString(fileContent.at(4), ",;:| \n\t").at(1));
        if (nbLayer < 2) {
            throw utils::TrackedException("Invalid number of layer: " + fileContent.at(4) + ".");
        }
        if (std::stoi(utils::splitString(fileContent.at(5), ",;:| \n\t").at(1)) < 1) {
            throw utils::TrackedException("Invalid number of layer: " + fileContent.at(5) + ".");
        }
        layerSize.push_back(std::stoul(utils::splitString(fileContent.at(5), ",;:| \n\t").at(1)));
    } catch (const std::exception &e) {
        throw utils::TrackedException("Invalid Hyper Parameter in " + filename + ".");
    }
    try {
        for (std::size_t i = 0; i < static_cast<std::size_t>(nbLayer - 1); ++i) {
            if (std::stoi(utils::splitString(fileContent.at(currentLine), ",;:| \n\t").at(1)) < 1) {
                throw utils::TrackedException("Invalid number of neuron: " + fileContent.at(currentLine) + ".");
            }
            layerSize.push_back(std::stoul(utils::splitString(fileContent.at(currentLine), ",;:| \n\t").at(1)));
            neuronLine.emplace_back(currentLine + 1, -1);
            while (!fileContent.at(currentLine).empty() &&
                   static_cast<std::size_t>(currentLine) < fileContent.size() - 1) {
                ++currentLine;
            }
            neuronLine.back().second = currentLine - 1;
            ++currentLine;
        }
    } catch (const std::exception &e) {
        throw utils::TrackedException("Invalid Layer size in " + filename + ".");
    }
    if ((nbInput > 0 && layerSize.front() != static_cast<std::size_t>(nbInput)) ||
        (nbOutput > 0 && layerSize.back() != static_cast<std::size_t>(nbOutput))) {
        throw utils::TrackedException("Invalid input/output layer size in " + filename + ".");
    }
    for (std::size_t i = 0; i < neuronLine.size(); ++i) {
        if (neuronLine[i].first >= neuronLine[i].second ||
            static_cast<std::size_t>(neuronLine[i].second - neuronLine[i].first) + 1 != layerSize.at(i + 1) * 4) {
            throw utils::TrackedException("Invalid neuron line in " + filename + ".");
        }
    }
    _createNeuron(layerSize);
    std::size_t neuronIt = 0;
    int j = 0;
    try {

        for (std::size_t i = 0; i < neuronLine.size(); ++i) {
            for (j = neuronLine[i].first; j <= neuronLine[i].second; j += 4) {
                _parseNeuronValueFromFile(
                    utils::splitString(fileContent.at(j), ",;:| \n\t"),
                    utils::splitString(fileContent.at(j + 1), ",;:| \n\t"),
                    utils::splitString(fileContent.at(j + 2), ",;:| \n\t"),
                    utils::splitString(fileContent.at(j + 3), ",;:| \n\t"),
                    layerSize,
                    std::make_pair(i + 1, neuronIt)
                );
                ++neuronIt;
            }
            neuronIt = 0;
        }
    } catch (const std::exception &e) {
        throw utils::TrackedException(
            "Invalid neuron value on line: " + std::to_string(j) + ": " + fileContent.at(j) +
            ", or: " + fileContent.at(j) + ", or: " + fileContent.at(j) + ", or: " + fileContent.at(j) + "."
        );
    }
}
