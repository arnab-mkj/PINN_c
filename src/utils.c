#include "utils.h"
#include <math.h>
#include <stdlib.h> // For rand()

double adaptive_learning_rate(double initial_rate, int epoch, double decay_rate) {
    if (decay_rate <= 0) return initial_rate; // Avoid division by zero or negative decay
    return initial_rate / (1.0 + decay_rate * epoch);
}

// Generates a random double between min and max
double random_double(double min, double max) {
    double scale = rand() / (double)RAND_MAX; // 0.0 to 1.0
    return min + scale * (max - min);         // min to max
}