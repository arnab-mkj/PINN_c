#ifndef LOSS_FUNCTIONS_H
#define LOSS_FUNCTIONS_H

#include <math.h>
#include "neural_network.h" // For NeuralNetwork and LossParameters type

// Constants (assuming some might be overridden by params if available)
#define ELECTRON_MASS 9.10938356e-31 // Electron mass (kg)
#define HBAR 1.0545718e-34          // Reduced Planck's constant (J·s)
#define EPSILON_0 8.854187817e-12    // Vacuum permittivity (F/m)
#define MU_0 1.2566370614e-6         // Vacuum permeability (N/A^2)
// K_B, E, ETA, G, RHO are less directly used in the revised PDE losses,
// but kept for potential use in other contexts or more detailed models.
#define K_B 1.380649e-23             // Boltzmann constant (J/K)
#define E 1.602176634e-19            // Elementary charge (C)
#define ETA 1.81e-5                  // Dynamic viscosity of air at room temperature (Pa·s)
#define G 9.81                       // Gravitational acceleration (m/s^2)
#define RHO_AIR 1.225                // Density of air at sea level (kg/m^3)

#define DEFAULT_RHO_CP 1.0e6 // Default (density * specific_heat_capacity) for heat equation if not specified

// Small value to prevent division by zero or instability
#define EPSILON 1e-10

// Forward declaration for NeuralNetwork if not included via neural_network.h
// typedef struct NeuralNetwork NeuralNetwork; - Handled by including neural_network.h
// typedef struct LossParameters LossParameters; - Handled by including neural_network.h
// typedef enum ActivationFunction ActivationFunction; - Handled by including neural_network.h


// Function declarations for PDE residual losses
// These now take the network, coordinates (x, t), spatial/temporal steps (dx, dt),
// and physical parameters. They compute the residual of the PDE.
// The NN output is typically output[0] for scalar fields.

double schrodinger_equation_residual(
    NeuralNetwork *nn,
    double x, double t,
    double dx, double dt,
    const LossParameters *params,
    ActivationFunction activation_func
);

double maxwell_equations_residual(
    NeuralNetwork *nn,
    double x, double t,
    double dx, double dt,
    const LossParameters *params,
    ActivationFunction activation_func
); // Assumes output[0] is E_z, output[1] is B_y for a 1D TE-like wave

double heat_equation_residual(
    NeuralNetwork *nn,
    double x, double t,
    double dx, double dt,
    const LossParameters *params,
    ActivationFunction activation_func
);

double wave_equation_residual(
    NeuralNetwork *nn,
    double x, double t,
    double dx, double dt,
    const LossParameters *params,
    ActivationFunction activation_func
);

// Boundary, initial, and conservation loss functions (signatures mostly unchanged)
// These typically compare the NN output at a specific point to a target value.
double boundary_condition_loss(double nn_output_at_boundary, double boundary_value);
double initial_condition_loss(double nn_output_at_initial_time, double initial_value);
double conservation_of_mass_loss(double divergence_velocity, double mass_source); // This one is more abstract in current context

// Adaptive normalization might be useful for combining different types of losses (PDE, BC, IC)
// For now, its direct use on single PDE residuals is removed.
double adaptive_normalization(const double *losses, int num_losses);

#endif // LOSS_FUNCTIONS_H