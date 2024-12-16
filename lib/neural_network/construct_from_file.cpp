/*
** EPITECH PROJECT, 2024
** B-CNA-500-PAR-5-1-neuralnetwork-thibaud.cathala
** File description:
** constructor
*/

#include <string>
#include "NeuralNetwork.hpp"
#include "TrackedException.hpp"

nn::NeuralNetwork::NeuralNetwork(const std::string &filename, int nbInput, int nbOutput)
{
    if (filename.ends_with(".nn")) {
        _createFromNN(filename, nbInput, nbOutput);
    } else if (nbInput == -1 && nbOutput == -1 && filename.ends_with(".conf")) {
        _createFromConf(filename);
    } else {
        throw utils::TrackedException("Invalid file extension: " + filename + ".");
    }
}
