/*
** EPITECH PROJECT, 2024
** B-CNA-500-PAR-5-1-neuralnetwork-thibaud.cathala
** File description:
** Metric
*/

#pragma once

#include <filesystem>
#include <fstream>
#include <string>

namespace train {

class Metric {
    private:
    std::ofstream _file;

    public:
    Metric(const std::string &destFilename)
    {
        _file.open(destFilename, std::ios::app);

        if (_file.is_open() && std::filesystem::is_empty(destFilename)) {
            _file << "Training Index;Train Accuracy Rate;Accuracy Rate;Success Rate;Learning Rate" << std::endl;
        }
    }

    void setMetric(std::size_t trainingI, double mse, double successRate, double accuracyRate, double learningRate)
    {
        if (_file.is_open()) {
            _file << trainingI << ';' << (1 - mse) * 100 << ';' << accuracyRate * 100 << ';' << successRate * 100 << ';'
                  << learningRate << std::endl;
        }
    }
};

} // namespace train
