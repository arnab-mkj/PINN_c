#include "loss_functions.h"
#include "neural_network.h" // For NeuralNetwork struct and forward_pass
#include <stdio.h> // For potential debugging
#include <string.h> // For strcmp if needed, though not directly here

// Helper to get NN output at a specific point (x,t)
// Output array must be pre-allocated (size OUTPUT_SIZE)
static void get_nn_output(NeuralNetwork *nn, double x, double t, double output_array[], ActivationFunction activation_func) {
    double nn_input[INPUT_SIZE] = {x, t};
    forward_pass(nn, nn_input, output_array, activation_func);
}

double adaptive_normalization(const double *losses, int num_losses) {
    double max_loss = 0.0;
    for (int i = 0; i < num_losses; i++) {
        if (fabs(losses[i]) > max_loss) {
            max_loss = fabs(losses[i]);
        }
    }
    return (max_loss > EPSILON) ? max_loss : EPSILON; // Avoid division by zero or very small numbers
}

// --- PDE Residual Functions ---
// These now calculate the residual of the PDE itself.
// The training loop will square this residual to form the loss.

double schrodinger_equation_residual(NeuralNetwork *nn, double x, double t, double dx, double dt, const LossParameters *params, ActivationFunction activation_func) {
    double psi_center[OUTPUT_SIZE], psi_t_plus_dt[OUTPUT_SIZE];
    double psi_x_plus_dx[OUTPUT_SIZE], psi_x_minus_dx[OUTPUT_SIZE];

    get_nn_output(nn, x, t, psi_center, activation_func);
    get_nn_output(nn, x, t + dt, psi_t_plus_dt, activation_func); // For time derivative
    get_nn_output(nn, x + dx, t, psi_x_plus_dx, activation_func); // For spatial derivative
    get_nn_output(nn, x - dx, t, psi_x_minus_dx, activation_func); // For spatial derivative

    // Using output[0] as the real part of psi for simplicity
    double psi = psi_center[0];
    double dpsi_dt_approx = (psi_t_plus_dt[0] - psi) / dt; // Forward difference for d(psi)/dt

    // Laplacian (d^2 psi / dx^2) using central difference
    double d2psi_dx2_approx = (psi_x_plus_dx[0] - 2.0 * psi + psi_x_minus_dx[0]) / (dx * dx);

    // Time-dependent Schrödinger equation (simplified, real form for i * dpsi/dt part)
    // i hbar d(psi)/dt = -hbar^2/(2m) * d^2(psi)/dx^2 + V * psi
    // We'll model a simplified version or one component.
    // For PINNs, often the complex residual is split into real and imaginary parts.
    // Here, let's assume psi is real and we are modeling something like:
    // hbar * d(psi)/dt_imaginary_part_proxy = -hbar^2/(2m) * d^2(psi)/dx^2 + V * psi
    // Or, if output[1] was psi_imaginary, we could do it properly.
    // For now, let's use a common form of the residual for a real field:
    // H_psi = -hbar^2/(2m) * d^2(psi)/dx^2 + V * psi
    // The loss would then be (H_psi - E*psi)^2 for eigenvalue problems, or include time derivative.
    // Let's use a simplified residual for demonstration:
    // residual = (HBAR * dpsi_dt_approx) - (-(HBAR * HBAR / (2.0 * ELECTRON_MASS)) * d2psi_dx2_approx + params->potential * psi);
    // This is tricky without complex numbers. Let's use the form from the paper often:
    // f_real = Re( i hbar psi_t - H psi) = -hbar psi_t_imag - (H psi_real)
    // f_imag = Im( i hbar psi_t - H psi) =  hbar psi_t_real - (H psi_imag)
    // Given we only have one output for psi (psi[0]), we'll simplify.
    // Let's assume output[0] is psi_real and output[1] is psi_imag
    // For now, stick to a simpler interpretation based on the original code's intent if psi is real:
    // The original loss had (psi - psi_target), kinetic_energy, potential_energy.
    // Let's compute a residual for:  hbar*dpsi/dt + (-hbar^2/2m)Laplacian(psi) + V*psi = 0
    // (assuming i was absorbed or we are looking at a specific component)

    double kinetic_term_val = -(HBAR * HBAR / (2.0 * ELECTRON_MASS)) * d2psi_dx2_approx;
    double potential_term_val = params->potential * psi;
    
    // Assuming output[0] is psi_real and output[1] is psi_imag
    // For simplicity, if only output[0] is used as a real wave function:
    // The time derivative term is often handled as (psi(t+dt) - psi(t-dt))/(2dt) for complex psi_t
    // Or, if we are solving for stationary states, dpsi_dt = -iE/hbar * psi
    // Let's use a simplified residual for a real field psi, focusing on spatial part and potential:
    // This is not the full Schrodinger, but reflects trying to minimize energy functional components.
    // A proper PINN for Schrodinger is more involved.
    // For now, let's use a placeholder residual that uses the terms:
    // This is a conceptual placeholder as the original loss was not a direct PDE residual.
    // A true residual for iħ∂ψ/∂t = Hψ would require complex arithmetic or splitting into real/imaginary parts.
    // Let's assume we are trying to satisfy: (-ħ²/2m)∇²ψ + Vψ = Eψ (time-independent)
    // and output[0] is ψ, and we are trying to make it an eigenfunction.
    // The 'E' (energy) would be another parameter or learned.
    // For now, let's use a simplified residual based on the terms:
    double residual = kinetic_term_val + potential_term_val; // This would be (H-E)psi if E was known/learned
                                                            // Or if time-dependent, dpsi_dt needs to be incorporated.
                                                            // The original loss was more like a sum of squared energy components.
                                                            // Let's try to make it closer to a PDE residual:
                                                            // Residual for: (-ħ²/2m)∇²ψ + Vψ - Eψ = 0
                                                            // Let's assume E is implicitly part of what the NN learns or is zero.
                                                            // For now, let's use a simplified form:
    residual = dpsi_dt_approx + kinetic_term_val + potential_term_val; // Simplified target = 0

    // Gradient penalty from original code (can be added to the squared residual later)
    // double gradient_penalty = 0.01 * pow(psi_x_plus_dx[0] - psi_x_minus_dx[0] / (2*dx) , 2); // Example gradient
    return residual;
}


double maxwell_equations_residual(NeuralNetwork *nn, double x, double t, double dx, double dt, const LossParameters *params, ActivationFunction activation_func) {
    // Assuming 1D Maxwell's equations (e.g., TEz mode: Ez, Hy)
    // ∂Ez/∂x = -μ₀ ∂Hy/∂t
    // ∂Hy/∂x = -ε₀ ∂Ez/∂t - Jz (if current Jz exists)
    // Let nn_output[0] be Ez, nn_output[1] be Hy
    double val_center[OUTPUT_SIZE], val_t_plus_dt[OUTPUT_SIZE], val_x_plus_dx[OUTPUT_SIZE];

    get_nn_output(nn, x, t, val_center, activation_func);
    get_nn_output(nn, x, t + dt, val_t_plus_dt, activation_func);
    get_nn_output(nn, x + dx, t, val_x_plus_dx, activation_func);

    double Ez = val_center[0];
    double Hy = val_center[1];

    double dEz_dx_approx = (val_x_plus_dx[0] - Ez) / dx; // Forward difference
    double dHy_dt_approx = (val_t_plus_dt[1] - Hy) / dt; // Forward difference

    double dHy_dx_approx = (val_x_plus_dx[1] - Hy) / dx; // Forward difference
    double dEz_dt_approx = (val_t_plus_dt[0] - Ez) / dt; // Forward difference

    double residual1 = dEz_dx_approx + MU_0 * dHy_dt_approx;
    double residual2 = dHy_dx_approx + EPSILON_0 * dEz_dt_approx + params->current_density; // Added Jz term
                                                                // If params->current_density is Jz:
                                                                // residual2 += params->current_density; (if current_density is Jz)


    // The original loss used Gauss's law and Ampere's law in a simplified form.
    // Gauss: electric_field - charge_density / EPSILON_0
    // Ampere: MU_0 * current_density - magnetic_field
    // This is hard to reconcile with a dynamic 1D wave.
    // For now, returning sum of squares of the 1D wave equation residuals.
    // The total residual could be sqrt(residual1^2 + residual2^2) or |residual1| + |residual2|
    // The training loop will square this. So we return a combined measure.
    return residual1 + residual2; // The training loop will square this. Or return one and train for both.
                                  // For multiple residuals, the loss function in training needs to handle it.
                                  // Let's return residual1 for now, assuming we focus on one part or combine later.
                                  // Or, the calling function expects a single value.
                                  // A common approach is sum of squared residuals: residual1^2 + residual2^2.
                                  // Since the training loop squares the output of this, we should return a single value.
                                  // Let's return (residual1 + residual2) / 2.0 or sqrt(res1^2+res2^2)
                                  // For now, let's return residual1 and acknowledge this is a simplification.
                                  // A better way: the main training loop calls this twice for res1, res2 or this returns an array.
                                  // Given current structure, let's return a single combined residual.
    return (residual1 + residual2); // This will be squared by the caller.
}


double heat_equation_residual(NeuralNetwork *nn, double x, double t, double dx, double dt, const LossParameters *params, ActivationFunction activation_func) {
    double u_center[OUTPUT_SIZE], u_t_plus_dt[OUTPUT_SIZE];
    double u_x_plus_dx[OUTPUT_SIZE], u_x_minus_dx[OUTPUT_SIZE];

    get_nn_output(nn, x, t, u_center, activation_func);
    get_nn_output(nn, x, t + dt, u_t_plus_dt, activation_func);
    get_nn_output(nn, x + dx, t, u_x_plus_dx, activation_func);
    get_nn_output(nn, x - dx, t, u_x_minus_dx, activation_func);

    double u = u_center[0];

    // du/dt using forward difference: (u(t+dt) - u(t)) / dt
    double du_dt_approx = (u_t_plus_dt[0] - u) / dt;

    // d^2u/dx^2 using central difference: (u(x+dx) - 2u(x) + u(x-dx)) / dx^2
    double d2u_dx2_approx = (u_x_plus_dx[0] - 2.0 * u + u_x_minus_dx[0]) / (dx * dx + EPSILON); // Added EPSILON to dx*dx

    // Thermal diffusivity alpha = k / (rho * c_p)
    // params->thermal_conductivity is k. We need rho_cp.
    double thermal_diffusivity = params->thermal_conductivity / (DEFAULT_RHO_CP + EPSILON); // Use a default rho_cp or pass it in params

    // PDE: du/dt = alpha * d^2u/dx^2  =>  du/dt - alpha * d^2u/dx^2 = 0
    double residual = du_dt_approx - thermal_diffusivity * d2u_dx2_approx;
    return residual;
}

double wave_equation_residual(NeuralNetwork *nn, double x, double t, double dx, double dt, const LossParameters *params, ActivationFunction activation_func) {
    double u_center[OUTPUT_SIZE];
    double u_t_plus_dt[OUTPUT_SIZE], u_t_minus_dt[OUTPUT_SIZE];
    double u_x_plus_dx[OUTPUT_SIZE], u_x_minus_dx[OUTPUT_SIZE];

    get_nn_output(nn, x, t, u_center, activation_func);
    get_nn_output(nn, x, t + dt, u_t_plus_dt, activation_func);
    get_nn_output(nn, x, t - dt, u_t_minus_dt, activation_func);
    get_nn_output(nn, x + dx, t, u_x_plus_dx, activation_func);
    get_nn_output(nn, x - dx, t, u_x_minus_dx, activation_func);

    double u = u_center[0];

    // d^2u/dt^2 using central difference: (u(t+dt) - 2u(t) + u(t-dt)) / dt^2
    double d2u_dt2_approx = (u_t_plus_dt[0] - 2.0 * u + u_t_minus_dt[0]) / (dt * dt + EPSILON);

    // d^2u/dx^2 using central difference: (u(x+dx) - 2u(x) + u(x-dx)) / dx^2
    double d2u_dx2_approx = (u_x_plus_dx[0] - 2.0 * u + u_x_minus_dx[0]) / (dx * dx + EPSILON);

    double c_squared = params->wave_speed * params->wave_speed;

    // PDE: d^2u/dt^2 = c^2 * d^2u/dx^2  =>  d^2u/dt^2 - c^2 * d^2u/dx^2 = 0
    double residual = d2u_dt2_approx - c_squared * d2u_dx2_approx;
    return residual;
}


// --- Boundary and Initial Condition Losses ---
// These remain largely the same, comparing a value to a target.
// The NORMALIZATION_FACTOR was removed from original h file, so using 1.0 or a small const.
#define BC_IC_NORMALIZATION_FACTOR 1.0 // Or some other suitable factor

double boundary_condition_loss(double nn_output_at_boundary, double boundary_value) {
    return pow(nn_output_at_boundary - boundary_value, 2) / BC_IC_NORMALIZATION_FACTOR;
}

double initial_condition_loss(double nn_output_at_initial_time, double initial_value) {
    return pow(nn_output_at_initial_time - initial_value, 2) / BC_IC_NORMALIZATION_FACTOR;
}

double conservation_of_mass_loss(double divergence_velocity, double mass_source) {
    // This function is quite abstract without a specific velocity field from the NN.
    // Assuming divergence_velocity is computed from NN outputs representing a velocity field.
    return pow(divergence_velocity - mass_source, 2) / BC_IC_NORMALIZATION_FACTOR;
}