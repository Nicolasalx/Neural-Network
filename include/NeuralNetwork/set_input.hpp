/*
** EPITECH PROJECT, 2024
** B-CNA-500-PAR-5-1-neuralnetwork-thibaud.cathala
** File description:
** set_input
*/

#pragma once

#include "NeuralNetwork.hpp"

namespace nn {

template <std::size_t NbInput>
void NeuralNetwork::_setInput(const std::array<bool, NbInput> &newInput)
{
    for (std::size_t i = 0; i < newInput.size(); ++i) {
        neuronValue.front().content[i] = newInput[i];
    }
}

}
