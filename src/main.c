#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h> // For srand

#include "neural_network.h"
// loss_functions.h and utils.h are included via neural_network.h or directly if needed

int main(int argc, char *argv[]) {
    srand(time(NULL)); // Seed random number generator once globally

    if (argc < 2) { // Simplified argument check, more robust parsing needed for all params
        fprintf(stderr, "Usage: %s --loss <type> --activation <function> [options...]\n", argv[0]);
        fprintf(stderr, "Required options for LossParameters (dx, dt, domain, etc.) are currently hardcoded or use defaults in train_neural_network.\n");
        fprintf(stderr, "Example: ./pinn --loss heat --activation tanh --epochs 10000 --lr 0.001 --potential 0.1 --charge_density 0 --current_density 0 --thermal_conductivity 0.05 --wave_speed 1.0\n");
        return EXIT_FAILURE;
    }

    char *loss_type = "heat"; // Default
    char *activation_function = "tanh"; // Default
    int epochs = 10000; // Default
    double learning_rate = 0.001; // Default

    // Initialize LossParameters with some defaults
    LossParameters params;
    params.potential = 0.1;
    params.charge_density = 0.0;
    params.current_density = 0.0;
    params.thermal_conductivity = 0.05; // Example value for k
    params.wave_speed = 1.0;

    // Default domain and discretization for training (can be overridden by more args)
    params.x_min = 0.0; params.x_max = 1.0;
    params.t_min = 0.0; params.t_max = 1.0;
    params.dx = 0.05;   // Spatial step for finite differences
    params.dt = 0.01;   // Time step for finite differences
    params.num_collocation_points = 100; // Number of random points per epoch

    // Basic command-line argument parsing (can be extended)
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--loss") == 0 && i + 1 < argc) {
            loss_type = argv[++i];
        } else if (strcmp(argv[i], "--activation") == 0 && i + 1 < argc) {
            activation_function = argv[++i];
        } else if (strcmp(argv[i], "--epochs") == 0 && i + 1 < argc) {
            epochs = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--lr") == 0 && i + 1 < argc) {
            learning_rate = atof(argv[++i]);
        } else if (strcmp(argv[i], "--potential") == 0 && i + 1 < argc) {
            params.potential = atof(argv[++i]);
        } else if (strcmp(argv[i], "--charge_density") == 0 && i + 1 < argc) {
            params.charge_density = atof(argv[++i]);
        } else if (strcmp(argv[i], "--current_density") == 0 && i + 1 < argc) {
            params.current_density = atof(argv[++i]);
        } else if (strcmp(argv[i], "--thermal_conductivity") == 0 && i + 1 < argc) {
            params.thermal_conductivity = atof(argv[++i]);
        } else if (strcmp(argv[i], "--wave_speed") == 0 && i + 1 < argc) {
            params.wave_speed = atof(argv[++i]);
        } else if (strcmp(argv[i], "--dx") == 0 && i + 1 < argc) {
            params.dx = atof(argv[++i]);
        } else if (strcmp(argv[i], "--dt") == 0 && i + 1 < argc) {
            params.dt = atof(argv[++i]);
        } else if (strcmp(argv[i], "--num_colloc") == 0 && i + 1 < argc) {
            params.num_collocation_points = atoi(argv[++i]);
        }
    }

    if (loss_type == NULL || activation_function == NULL) {
        fprintf(stderr, "Error: Loss type or activation function not specified correctly.\n");
        return EXIT_FAILURE;
    }
     if (params.dx <= 0 || params.dt <= 0) {
        fprintf(stderr, "Error: dx and dt must be positive.\n");
        return EXIT_FAILURE;
    }
    if (params.num_collocation_points <=0) {
        fprintf(stderr, "Error: Number of collocation points must be positive.\n");
        return EXIT_FAILURE;
    }


    NeuralNetwork nn;
    initialize_neural_network(&nn);
    if (!validate_neural_network_initialization(&nn)) { // Basic validation
        fprintf(stderr, "Neural Network initialization failed validation (if any checks implemented).\n");
        // return EXIT_FAILURE; // Not critical if validation is basic
    }

    train_neural_network(&nn, loss_type, &params, epochs, learning_rate, activation_function);

    save_model(&nn, "pinn_model_trained.txt");

    return EXIT_SUCCESS;
}