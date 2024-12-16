/*
** EPITECH PROJECT, 2024
** B-CNA-500-PAR-5-1-neuralnetwork-thibaud.cathala
** File description:
** printProgress
*/

#include <iomanip>
#include <iostream>
#include "NeuralNetwork.hpp"

#ifndef STATIC_NEURAL_NETWORK
static void drawBar(double progressLevel, int barSize, double medium, double high)
{
    int progress = 0;
    std::string bar;

    progress = static_cast<int>(std::round(progressLevel * barSize));
    if (progressLevel > high) {
        bar += "\e[92m";
    } else if (progressLevel > medium) {
        bar += "\e[93m";
    } else {
        bar += "\e[91m";
    }
    for (int i = 0; i < progress; ++i) {
        bar += "█";
    }
    bar += "\e[0m";
    for (int i = 0; i < barSize - progress; ++i) {
        bar += "▒";
    }
    std::cout << "[" << bar << "] " << std::setw(10) << std::setprecision(6) << std::fixed << progressLevel * 100.0
              << "%";
}

static void drawProgressBar(double progressLevel, int barSize)
{
    int progress = 0;
    std::string bar;

    progress = static_cast<int>(std::round(progressLevel * barSize));
    bar += "\e[2m";
    for (int i = 0; i < progress; ++i) {
        bar += "█";
    }
    for (int i = 0; i < barSize - progress; ++i) {
        bar += "▒";
    }
    std::cout << "[" << bar << "]\e[0m " << std::setw(6) << std::setprecision(2) << std::fixed << progressLevel * 100.0
              << "%";
}

void nn::NeuralNetwork::printProgress(double successRate, double accuracyRate, std::size_t trainingProgress, std::size_t batch, int barSize)
    const
{
    for (const auto &it : neuronValue) {
        std::cout << it.content.size() << " ";
    }
    std::cout << "| Progress:";
    drawProgressBar(static_cast<double>(trainingProgress) / static_cast<double>(batch), barSize);
    std::cout << ", Success Rate:";
    drawBar(successRate, barSize, 0.70, 0.90);
    std::cout << ", Accuracy Rate:";
    drawBar(accuracyRate, barSize, 0.70, 0.90);
    std::cout << ", learning Rate:" << learningRate << "\n";
}
#endif
