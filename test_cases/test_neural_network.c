#include <stdio.h>
#include <stdlib.h> // For srand
#include <time.h>   // For time
#include "../include/neural_network.h" // Make sure path is correct
#include "../include/utils.h"          // For random_double if used directly

void test_initialize_neural_network() {
    NeuralNetwork nn;
    initialize_neural_network(&nn);
    // A simple visual check or more complex statistical check could be done.
    // For now, just confirm it runs.
    if (validate_neural_network_initialization(&nn)) { // Assuming validate is very basic
        printf("Neural Network Initialized (validation is basic).\n");
        // Example: print one weight to see if it's non-zero
        printf("  Example weight nn.weights_input_hidden[0][0]: %lf\n", nn.weights_input_hidden[0][0]);
    } else {
        printf("Neural Network Initialization Validation Failed.\n");
    }
}

void test_forward_pass() {
    NeuralNetwork nn;
    initialize_neural_network(&nn); // Initialize with random weights

    double input[INPUT_SIZE] = {0.5, 0.2}; // Example input (x, t)
    double output[OUTPUT_SIZE];
    ActivationFunction act_func = TANH; // Example

    forward_pass(&nn, input, output, act_func);

    printf("Forward Pass Test with %s:\n", (act_func == TANH ? "TANH" : (act_func == SIGMOID ? "SIGMOID" : (act_func == RELU ? "RELU" : "LEAKY_RELU"))));
    printf("  Input: ");
    for(int i=0; i<INPUT_SIZE; ++i) printf("%.2f ", input[i]);
    printf("\n");
    printf("  Output: ");
    for (int k = 0; k < OUTPUT_SIZE; k++) {
        printf("%.4f ", output[k]);
    }
    printf("\n");
}

int main() {
    srand(time(NULL)); // Seed random numbers

    printf("--- Testing Neural Network Initialization ---\n");
    test_initialize_neural_network();

    printf("\n--- Testing Neural Network Forward Pass ---\n");
    test_forward_pass();

    return 0;
}