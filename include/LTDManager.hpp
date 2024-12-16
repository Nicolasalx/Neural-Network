/*
** EPITECH PROJECT, 2024
** B-CNA-500-PAR-5-1-neuralnetwork-thibaud.cathala
** File description:
** LargeFileManager
*/

#pragma once

#include <algorithm>
#include <fstream>
#include <functional>
#include <string>
#include <vector>
#include "TrackedException.hpp"
#include "TrainingData.hpp"

class LTDManager {
    private:
    std::size_t _trainingDataChunkSize = 10'000;
    std::size_t _nbChunkRetreived = 0;
    std::ifstream _file;
    bool _allDataRetrieved = false;
    std::size_t _nbLine = 0;

    std::size_t _countNbLine(const std::string &filename)
    {
        std::ifstream file(filename);

        if (!file.is_open()) {
            throw std::runtime_error("Cannot open file: " + filename);
        }
        return std::count(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>(), '\n') + 1;
    }

    public:
    LTDManager(const std::string &filename, std::size_t trainingDataChunkSize)
        : _trainingDataChunkSize(trainingDataChunkSize)
    {
        _file.open(filename);
        if (!_file.is_open()) {
            throw utils::TrackedException("Fail to open file: " + filename + ".");
        }
        _nbLine = _countNbLine(filename);
    }

    double getNbChunkRetreived() const
    {
        return static_cast<double>(_nbChunkRetreived);
    }

    template <std::size_t NbInput, std::size_t NbOutput>
    std::vector<nn::TrainingData<NbInput, NbOutput>> getEntireTrainingData(
        const std::function<nn::TrainingData<NbInput, NbOutput>(const std::string &)> &parsingFunc
    )
    {
        std::vector<nn::TrainingData<NbInput, NbOutput>> trainingData;
        std::string line;

        if (_allDataRetrieved) {
            return {};
        }
        if (!_file.is_open()) {
            throw utils::TrackedException("Fail to read file content.");
        }
        trainingData.reserve(_nbLine);
        while (std::getline(_file, line)) {
            trainingData.push_back(parsingFunc(line));
        }
        _allDataRetrieved = true;
        ++_nbChunkRetreived;
        return trainingData;
    }

    template <
        std::size_t NbInput,
        std::size_t NbOutput,
        std::function<nn::TrainingData<NbInput, NbOutput>(const std::string &)> ParsingFunc>
    void getTrainingData(const std::function<void(const std::vector<nn::TrainingData<NbInput, NbOutput>> &)> &func)
    {
        std::vector<nn::TrainingData<NbInput, NbOutput>> trainingData;
        std::string line;

        if (_allDataRetrieved) {
            return;
        }
        trainingData.reserve(_trainingDataChunkSize);
        while (true) {
            if (!_file.is_open()) {
                throw utils::TrackedException("Fail to read file content.");
            }
            for (std::size_t i = 0; i < _trainingDataChunkSize; ++i) {
                if (!std::getline(_file, line)) {
                    _allDataRetrieved = true;
                    func(trainingData);
                    trainingData.clear();
                    ++_nbChunkRetreived;
                    return;
                }
                trainingData.push_back(ParsingFunc(line));
            }
            func(trainingData);
            trainingData.clear();
            ++_nbChunkRetreived;
        }
    }
};
