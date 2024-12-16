/*
** EPITECH PROJECT, 2024
** B-CNA-500-PAR-5-1-neuralnetwork-thibaud.cathala
** File description:
** compute_neuron_value
*/

#include <immintrin.h>
#include "NeuralNetwork.hpp"

#ifndef STATIC_NEURAL_NETWORK
void nn::NeuralNetwork::_computeNeuronValue(std::size_t layerIt, std::size_t neuronIt)
{
// #if (defined( __AVX512F__) && !defined(CHECK_MEMORY))
//     double sigma = neuronBias[layerIt].content[neuronIt];
//     constexpr std::size_t VECTOR_SIZE = sizeof(__m512d) / sizeof(double);
//     __m512d sum = _mm512_setzero_pd();
//     std::size_t i = 0;

//     for (i = 0; i + VECTOR_SIZE <= neuronWeight[layerIt].lineSize; i += VECTOR_SIZE) {
//         sum = _mm512_fmadd_pd(
//             _mm512_loadu_pd(&neuronValue[layerIt - 1].content[i]),
//             _mm512_loadu_pd(&neuronWeight[layerIt](neuronIt, i)),
//             sum
//         );
//     }
//     sigma += _mm512_reduce_add_pd(sum);
//     for (; i < neuronWeight[layerIt].lineSize; ++i) {
//         sigma += neuronValue[layerIt - 1].content[i] * neuronWeight[layerIt](neuronIt, i);
//     }
//     neuronValue[layerIt].content[neuronIt] = _sigmoid(sigma);
// #else
    double sigma = neuronBias[layerIt].content[neuronIt];

    for (std::size_t weightIt = 0; weightIt < neuronWeight[layerIt].lineSize; ++weightIt) {
        sigma += neuronValue[layerIt - 1].content[weightIt] * neuronWeight[layerIt](neuronIt, weightIt);
    }
    neuronValue[layerIt].content[neuronIt] = _sigmoid(sigma);
// #endif
}
#endif
