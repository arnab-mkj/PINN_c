#ifndef UTILS_H
#define UTILS_H

#include <stddef.h> // For size_t

// Adaptive learning rate (remains useful)
double adaptive_learning_rate(double initial_rate, int epoch, double decay_rate);

// Utility for random number generation in a range (optional, but good for collocation points)
double random_double(double min, double max);

#endif // UTILS_H