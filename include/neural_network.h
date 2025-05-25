#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#define INPUT_SIZE 2  // Typically (x, t)
#define HIDDEN_SIZE 5 // Kept small for simplicity
#define OUTPUT_SIZE 3 // output[0] for scalar PDEs, more for vector (e.g., Maxwell)

typedef struct {
    double weights_input_hidden[INPUT_SIZE][HIDDEN_SIZE];
    double weights_hidden_output[HIDDEN_SIZE][OUTPUT_SIZE];
    double biases_hidden[HIDDEN_SIZE];
    double biases_output[OUTPUT_SIZE];
    double hidden_outputs[HIDDEN_SIZE]; // To store hidden layer activations for backprop
} NeuralNetwork;

typedef enum {
    RELU,
    SIGMOID,
    TANH,
    LEAKY_RELU
} ActivationFunction;

// Parameters for the physical problem and training
typedef struct {
    double potential;          // For Schrödinger
    double charge_density;     // For Maxwell
    double current_density;    // For Maxwell
    double thermal_conductivity; // For Heat Equation
    double wave_speed;         // For Wave Equation
    // Domain and discretization parameters for training
    double x_min, x_max;
    double t_min, t_max;
    double dx;                 // Spatial step for finite differences
    double dt;                 // Time step for finite differences
    int num_collocation_points;
} LossParameters; // Renamed from PDEParameters for clarity, also includes training params

void initialize_neural_network(NeuralNetwork *nn);
int validate_neural_network_initialization(const NeuralNetwork *nn);

// Forward pass: input is typically (x,t), output is the NN's prediction u(x,t)
void forward_pass(NeuralNetwork *nn, double input[INPUT_SIZE], double output[OUTPUT_SIZE], ActivationFunction activation_function);

// Training function
void train_neural_network(
    NeuralNetwork *nn,
    const char *loss_type_str, // String name of the loss (e.g., "schrodinger")
    LossParameters *params,    // Includes physical and training domain parameters
    int epochs,
    double learning_rate,
    const char *activation_function_str // String name of activation
);

void save_model(const NeuralNetwork *nn, const char *filename);

// Helper to get activation function enum from string
ActivationFunction get_activation_function_from_string(const char *activation_str);


#endif // NEURAL_NETWORK_H