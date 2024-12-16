/*
** EPITECH PROJECT, 2024
** B-CNA-500-PAR-5-1-neuralnetwork-thibaud.cathala
** File description:
** main
*/

#include <exception>
#include <format>
#include <string>
#include "Logger.hpp"
#include "NeuralNetwork.hpp"
#include "TrackedException.hpp"

int main(int argc, const char *argv[])
{
    int nbNN = 0;

    if (((argc - 1) % 2) != 0 || argc == 1) {
        utils::logError(std::format("Invalid number of arg: {}.", argc));
        return 84;
    }
    for (int i = 1; i < argc; i += 2) {
        try {
            if (!std::string(argv[i]).ends_with(".conf") || std::string(argv[i]).size() < 5) {
                throw utils::TrackedException(
                    "Invalid config file name: " + std::string(argv[i]) + ", it must containt the .conf extention."
                );
            }
            nbNN = std::stoi(argv[i + 1]);
            for (int j = 0; j < nbNN; ++j) {
                nn::NeuralNetwork(argv[i], -1, -1).encode(
                    std::format("{}_{}.nn", std::string(argv[i]).substr(0, std::string(argv[i]).size() - 5), j + 1)
                );
            }
        } catch (const std::exception &e) {
            utils::logWarning(e.what());
        }
    }
    return 0;
}
