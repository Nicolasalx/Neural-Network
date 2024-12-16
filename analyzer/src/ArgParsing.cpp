/*
** EPITECH PROJECT, 2024
** B-CNA-500-PAR-5-1-neuralnetwork-thibaud.cathala
** File description:
** ArgParsing
*/

#include "ArgParsing.hpp"
#include <iostream>
#include "TrackedException.hpp"

void ArgParsing::printHelp()
{
    std::cout << "Usage: ./my_torch_analyzer [--predict | --train [--save SAVEFILE]] LOADFILE FILE\n"
                 "       ./my_torch_analyzer --predict LOADFILE FILE\n"
                 "       ./my_torch_analyzer --predict --color LOADFILE FILE\n"
                 "       ./my_torch_analyzer --train LOADFILE FILE\n"
                 "       ./my_torch_analyzer --train [--save SAVEFILE] LOADFILE FILE\n"
                 "       ./my_torch_analyzer --train_and_evaluate SAVEFILE TRAINING_FILE EVALUATE_FILE\n";
}

void ArgParsing::registerOption(const std::string &name, const std::function<void(int, const char **)> &func)
{
    _option[name] = func;
}

void ArgParsing::handleArg(int argc, const char **argv)
{
    if (argc < 2) {
        throw utils::TrackedException("Invalid number of arguments.");
    }
    if (_option.contains(argv[1])) {
        _option.at(std::string(argv[1]))(argc, argv);
        return;
    }
    throw utils::TrackedException("Invalid number of arguments.");
}
