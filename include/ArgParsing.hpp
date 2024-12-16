/*
** EPITECH PROJECT, 2024
** B-CNA-500-PAR-5-1-neuralnetwork-thibaud.cathala
** File description:
** ArgParsing
*/

#pragma once

#include <functional>
#include <string>
#include <unordered_map>

class ArgParsing {
    std::unordered_map<std::string, std::function<void(int, const char **)>> _option;

    public:
    ArgParsing() = default;

    void printHelp();
    void registerOption(const std::string &name, const std::function<void(int, const char **)> &func);
    void handleArg(int argc, const char **argv);
};
