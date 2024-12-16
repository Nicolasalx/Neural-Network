/*
** EPITECH PROJECT, 2024
** B-CNA-500-PAR-5-1-neuralnetwork-thibaud.cathala
** File description:
** TrainingData
*/

#pragma once

#include <array>

namespace nn {

template <std::size_t NbInput, std::size_t NbOutput>
using TrainingData = std::pair<std::array<bool, NbInput>, std::array<bool, NbOutput>>;

}
