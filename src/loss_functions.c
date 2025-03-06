#include "loss_functions.h"
#include <math.h>
#include <stdio.h>

double adaptive_normalization(double *losses, int num_losses){
    double max_loss= 0.0;
    for (int i=0; i < num_losses; i++){
        if (losses[i] > max_loss){
            max_loss = losses[i];
        }
    }
    return (max_loss > 1.0e-10) ? max_loss : 1.0e-10;
}

double schrodinger_equation_loss(double psi, double psi_target, double potential, double time_step){
    double difference = psi - psi_target;
    double kinetic_energy = -(HBAR * HBAR / (2.0 * ELECTRON_MASS)) * (difference / (time_step * time_step));
    double potential_energy = potential *psi;

    double loss = (pow(difference, 2) + pow(kinetic_energy, 2) + pow(potential_energy, 2)) / (adaptive_normalization(&loss, 1)); //higher-order term for stability

    double gradient_penalty = 0.01 * (pow(psi - psi_target, 2)); // penalty for large gradients to smooth the loss landscape

    return loss + gradient_penalty;
}

double maxwell_equations_loss(double electric_field, double magnetic_field, double charge_density, double current_density) {
    // Compute residuals for Gauss's law and Ampere's law
    double gauss_residual = electric_field - charge_density / EPSILON_0; // Gauss's law
    double ampere_residual = MU_0 * current_density - magnetic_field;    // Ampere's law

    // Calculate squared residuals
    double gauss_loss = pow(gauss_residual, 2);
    double ampere_loss = pow(ampere_residual, 2);

    // Total loss calculation
    double total_loss = (gauss_loss + ampere_loss);

    return total_loss;
}

double heat_equation_loss(double u, double u_target, double dx, double dt){
    double thermal_diffusivity = K_B / (ETA * ELECTRON_MASS);

    double time_derivative = (u - u_target) / dt; //first derivative wrt time
    double second_spatial_derivative = (u - 2 * u_target + (u - u_target)) / (dx * dx); // second derivative wrt space

    double loss = pow(thermal_diffusivity * second_spatial_derivative - time_derivative, 2) / (adaptive_normalization(&loss, 1));
     return loss;
}

double wave_equation_loss(double u, double u_target, double time, double dx, double dt){
    (void)time;
    double spatial_term = (u-u_target)/ (dx * dx);
    double temporal_term = (u - u_target) / (dt * dt);

    double loss = pow((1.0 / (WAVE_SPEED * WAVE_SPEED)) * spatial_term - temporal_term, 2) / (adaptive_normalization(&loss, 1));
     return loss;
}

double boundary_condition_loss(double value, double boundary_value){
    return pow(value - boundary_value, 2) / NORMALIZATION_FACTOR;
}

double initial_condition_loss(double value, double initial_value){
    return pow(value - initial_value, 2) / NORMALIZATION_FACTOR;
}

double conservation_of_mass_loss(double divergence_velocity, double mass_source){
    return pow(divergence_velocity - mass_source, 2) / NORMALIZATION_FACTOR;
}
