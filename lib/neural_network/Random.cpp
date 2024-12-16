/*
** EPITECH PROJECT, 2024
** B-CNA-500-PAR-5-1-neuralnetwork-thibaud.cathala
** File description:
** Random
*/

#include "Random.hpp"

std::random_device nn::Random::rd;
std::mt19937 nn::Random::gen(rd());
std::uniform_real_distribution<double> nn::Random::dist(-1, 1);

double nn::Random::normalized()
{
    return dist(gen);
}

std::size_t nn::Random::range(std::size_t min, std::size_t max)
{
    std::uniform_int_distribution<std::size_t> intDist(min, max);
    return intDist(gen);
}
