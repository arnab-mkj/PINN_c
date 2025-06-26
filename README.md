# Physics-Informed Neural Network (PINN) in C for Solving PDEs

## Overview

This project implements a **Physics-Informed Neural Network (PINN)** in C to approximate solutions for various partial differential equations (PDEs) relevant to computational physics and chemistry. These include the Schrödinger equation, Maxwell’s equations (simplified 1D), the heat equation, and the wave equation.

PINNs are a class of neural networks that embed knowledge of physical laws, described by PDEs, directly into the learning process. Instead of solely relying on data, the network is trained to minimize a loss function that includes the residual of the PDE itself. This allows the network to learn solutions that are consistent with the underlying physics, even with sparse data.

This C implementation focuses on a feedforward neural network architecture. The training process involves sampling collocation points within the spatio-temporal domain of the PDE and adjusting the network's weights and biases to minimize the PDE residuals at these points.

This project demonstrates concepts in:

-   **Computational Physics/Chemistry**: Approximating solutions to fundamental PDEs.
-   **Numerical Methods**: Using finite difference approximations for derivative terms within the loss function.
-   **Machine Learning**: Implementing and training a neural network from scratch.
-   **C Programming**: Modular code structure with a focus on understanding core mechanisms.

## Features

-   **Feedforward Neural Network**:
    -   Configurable input size (typically 2 for `(x, t)` coordinates).
    -   Configurable hidden layer size.
    -   Configurable output size (typically 1 for scalar PDEs, or more for vector fields like Maxwell's).
    -   Supports multiple activation functions for the hidden layer: ReLU, Sigmoid, Tanh, Leaky ReLU.
    -   Linear activation for the output layer (common for regression and PDE solving).
-   **Physics-Informed Loss Functions**:
    -   The core loss is derived from the residual of the target PDE. The network is trained to make this residual approach zero.
    -   Implemented for:
        -   **Schrödinger Equation** (simplified, real-valued, time-dependent aspects)
        -   **Maxwell’s Equations** (simplified 1D formulation, e.g., TEz mode `Ez, Hy`)
        -   **Heat Equation** (1D diffusion)
        -   **Wave Equation** (1D wave propagation)
    -   Derivative terms in PDEs are approximated using finite differences (central, forward).
-   **Training Process**:
    -   Uses stochastic gradient descent (SGD) with backpropagation to update network weights.
    -   Samples random **collocation points** from the defined spatio-temporal domain (`x_min, x_max, t_min, t_max`) for training.
    -   Supports adaptive learning rate decay.
    -   Command-line configuration for PDE type, activation function, training epochs, learning rate, and PDE-specific physical parameters.
-   **Logging**: Saves training progress (epoch, average loss) to a CSV-formatted log file for analysis and plotting.
-   **Model Saving**: Saves the trained neural network parameters (weights and biases) to a text file.
-   **Testing**: Includes basic test cases for PDE residual functions and neural network operations.




## The Neural Network Model

The core of the PINN is a simple feedforward neural network (also known as a Multi-Layer Perceptron - MLP).

-   **Inputs**: The network takes `INPUT_SIZE` inputs. For solving PDEs, these are typically the independent variables of the PDE, such as spatial coordinate `x` and time `t`. So, `INPUT_SIZE` is often 2.
-   **Hidden Layer**: There is one hidden layer with `HIDDEN_SIZE` neurons. Each neuron in the hidden layer computes a weighted sum of the inputs, adds a bias, and then applies an activation function (ReLU, Sigmoid, Tanh, or Leaky ReLU). The choice of activation function can impact the network's ability to approximate complex solutions.
    -   Neuron output: `h_j = activation(sum(w_ij * x_i) + b_j)`
-   **Output Layer**: The hidden layer outputs are then fed into an output layer with `OUTPUT_SIZE` neurons. For scalar PDEs (like the heat or wave equation where the solution `u(x,t)` is a single value), `OUTPUT_SIZE` is typically 1. For vector PDEs or systems of equations (like Maxwell's equations representing `Ez` and `Hy`), `OUTPUT_SIZE` can be greater than 1. The output layer in this implementation uses a linear activation function (i.e., identity function), which is common when the network is directly approximating the value of a physical field.
    -   Network output: `NN_k(x,t) = sum(w_jk * h_j) + b_k`
-   **Parameters**: The learnable parameters of the network are its weights (`w`) and biases (`b`) for both the input-to-hidden and hidden-to-output layers.

The neural network `NN(x,t; θ)` (where `θ` represents all weights and biases) learns to approximate the solution of the PDE, e.g., `u_approx(x,t) = NN(x,t; θ)`.

## Method: Solving PDEs with PINNs

The fundamental idea is to train the neural network `NN(x,t; θ)` such that its output, when plugged into the PDE, makes the PDE residual close to zero.

Consider a general PDE:
$$
\mathcal{F}(u, \frac{\partial u}{\partial t}, \frac{\partial u}{\partial x}, \frac{\partial^2 u}{\partial x^2}, \dots; \lambda) = 0 \quad \text{for } (x,t) \in \Omega
$$
where `u(x,t)` is the unknown solution, `λ` represents physical parameters, and `Ω` is the domain.

The PINN approximates `u(x,t)` with `NN(x,t; θ)`. The PDE residual is then:
$$
R(x,t; \theta) = \mathcal{F}(NN, \frac{\partial NN}{\partial t}, \frac{\partial NN}{\partial x}, \frac{\partial^2 NN}{\partial x^2}, \dots; \lambda)
$$

The **loss function** for the PDE part is typically the mean squared error of this residual over a set of sampled collocation points `(x_i, t_i)` from the domain `Ω`:
$$
L_{PDE}(\theta) = \frac{1}{N_c} \sum_{i=1}^{N_c} |R(x_i, t_i; \theta)|^2
$$
where `N_c` is the number of collocation points.

**Derivatives**: The derivatives of the neural network output `NN` with respect to its inputs (`x`, `t`) are needed to compute the residual `R`. In this C implementation, these derivatives are approximated using **finite difference methods**. For example:
-   First derivative (forward difference): `∂NN/∂t ≈ (NN(x, t+Δt) - NN(x,t)) / Δt`
-   Second derivative (central difference): `∂²NN/∂x² ≈ (NN(x+Δx, t) - 2NN(x,t) + NN(x-Δx, t)) / (Δx)²`
More advanced PINN frameworks often use automatic differentiation for exact derivatives, but that is more complex to implement from scratch.

**Training**:
1.  Initialize the neural network weights `θ` randomly.
2.  For each training epoch:
    a.  Sample a batch of `N_c` collocation points `(x_j, t_j)` from the domain `Ω`.
    b.  For each collocation point:
        i.  Compute `NN(x_j, t_j; θ)`.
        ii. To compute derivatives, also compute `NN` at neighboring points (e.g., `NN(x_j+Δx, t_j)`, `NN(x_j, t_j+Δt)` etc.).
        iii. Calculate the PDE residual `R(x_j, t_j; θ)` using these values and finite differences.
    c.  Calculate the total loss `L_PDE(θ)`. (Boundary and initial condition losses would also be added here if fully implemented).
    d.  Compute the gradients of the loss with respect to the network parameters `θ` (i.e., `∂L/∂θ`) using the backpropagation algorithm.
    e.  Update the parameters `θ` using an optimization algorithm like stochastic gradient descent (SGD): `θ_new = θ_old - learning_rate * ∂L/∂θ`.
3.  Repeat for a specified number of epochs.

As training progresses, the network `NN(x,t; θ)` should evolve to become a good approximation of the true solution `u(x,t)`.

### Example: Solving the 1D Heat Equation

The 1D heat equation is:
$$
\frac{\partial u}{\partial t} = \alpha \frac{\partial^2 u}{\partial x^2}
$$
where `u(x,t)` is the temperature and `α` is the thermal diffusivity (`α = k / (ρ * c_p)` where `k` is thermal conductivity).

The PDE residual function `R(x,t; θ)` for the neural network `NN(x,t; θ)` is:
$$
R(x,t; \theta) = \frac{\partial NN}{\partial t} - \alpha \frac{\partial^2 NN}{\partial x^2}
$$

In `src/loss_functions.c`, `heat_equation_residual` calculates this:
1.  It takes the current network `nn`, coordinates `(x,t)`, step sizes `dx, dt`, physical parameters `params` (which includes `params->thermal_conductivity`), and the `activation_func`.
2.  It calls `forward_pass` (via `get_nn_output`) to get:
    -   `NN(x,t)` (denoted `u` or `u_center[0]`)
    -   `NN(x, t+dt)` (for `u_t_plus_dt[0]`)
    -   `NN(x+dx, t)` (for `u_x_plus_dx[0]`)
    -   `NN(x-dx, t)` (for `u_x_minus_dx[0]`)
3.  It approximates the derivatives:
    -   `∂NN/∂t ≈ (u_t_plus_dt[0] - u_center[0]) / dt`
    -   `∂²NN/∂x² ≈ (u_x_plus_dx[0] - 2*u_center[0] + u_x_minus_dx[0]) / (dx*dx)`
4.  It calculates thermal diffusivity `α` using `params->thermal_conductivity` and a default `DEFAULT_RHO_CP`.
5.  It returns the residual: `residual = (∂NN/∂t) - α * (∂²NN/∂x²)`.

During training (`train_neural_network` in `src/neural_network.c`):
-   For each collocation point `(x_colloc, t_colloc)`:
    -   The `heat_equation_residual` is called.
    -   The loss for this point is `0.5 * residual^2`.
    -   `update_weights_pinn` performs backpropagation to adjust network weights to reduce this squared residual.

## PDE Residual Functions Explained

Located in `src/loss_functions.c`. These functions compute the value `R` that should ideally be zero if the NN output perfectly satisfies the PDE. The training loop then squares this residual to form the loss.

#### `schrodinger_equation_residual`
-   **Equation (Conceptual Target)**: A simplified, real-valued form related to the time-dependent Schrödinger equation `iħ∂ψ/∂t = Hψ`, where `H = (-ħ²/2m)∇² + V`. Due to the complexity of `i` and potentially complex `ψ`, this implementation models a simplified residual.
    $$ R = \frac{\partial \psi_{NN}}{\partial t} + \frac{1}{\hbar} \left( \frac{-\hbar^2}{2m} \frac{\partial^2 \psi_{NN}}{\partial x^2} + V \psi_{NN} \right) $$
    (This is one possible interpretation for a real `ψ` and aiming for `R=0`. A full treatment is more complex.)
-   **Implementation**:
    -   `ψ_NN` is `nn->output[0]`.
    -   `∂ψ_NN/∂t` approximated by forward difference.
    -   `∂²ψ_NN/∂x²` approximated by central difference.
    -   `V` is `params->potential`.
    -   Constants `ħ` (HBAR) and `m` (ELECTRON_MASS) are used.

#### `maxwell_equations_residual`
-   **Equations (1D TE-like mode)**:
    $$\frac{\partial E_z}{\partial x} = -\mu_0 \frac{\partial H_y}{\partial t} \implies R_1 = \frac{\partial E_z}{\partial x} + \mu_0 \frac{\partial H_y}{\partial t} $$
    $$ \frac{\partial H_y}{\partial x} = -\epsilon_0 \frac{\partial E_z}{\partial t} - J_z \implies R_2 = \frac{\partial H_y}{\partial x} + \epsilon_0 \frac{\partial E_z}{\partial t} + J_z $$
-   **Implementation**:
    -   Assumes `NN_output[0]` is `E_z` and `NN_output[1]` is `H_y`.
    -   Derivatives are approximated using forward differences.
    -   `J_z` is `params->current_density`.
    -   Constants `μ₀` (MU_0) and `ε₀` (EPSILON_0) are used.
    -   The function returns `R_1 + R_2`. The training loop squares this sum. For a more rigorous approach, the loss would be `R_1^2 + R_2^2`. The `pde_residuals_vector` in `train_neural_network` is set up to handle this if `maxwell_equations_residual` were to fill its components, but currently, it uses the summed scalar.

#### `heat_equation_residual`
-   **Equation (1D)**:
    $$ \frac{\partial u}{\partial t} = \alpha \frac{\partial^2 u}{\partial x^2} \implies R = \frac{\partial u}{\partial t} - \alpha \frac{\partial^2 u}{\partial x^2} $$
-   **Implementation**:
    -   `u_NN` is `nn->output[0]`.
    -   `∂u_NN/∂t` approximated by forward difference.
    -   `∂²u_NN/∂x²` approximated by central difference.
    -   `α` (thermal_diffusivity) is calculated from `params->thermal_conductivity` and `DEFAULT_RHO_CP`.

#### `wave_equation_residual`
-   **Equation (1D)**:
    $$ \frac{\partial^2 u}{\partial t^2} = c^2 \frac{\partial^2 u}{\partial x^2} \implies R = \frac{\partial^2 u}{\partial t^2} - c^2 \frac{\partial^2 u}{\partial x^2} $$
-   **Implementation**:
    -   `u_NN` is `nn->output[0]`.
    -   `∂²u_NN/∂t²` approximated by central difference.
    -   `∂²u_NN/∂x²` approximated by central difference.
    -   `c` is `params->wave_speed`.

### Other Loss Functions
-   **`boundary_condition_loss(nn_output, boundary_value)`**: Calculates `(nn_output - boundary_value)^2`. Used to enforce Dirichlet boundary conditions.
-   **`initial_condition_loss(nn_output, initial_value)`**: Calculates `(nn_output - initial_value)^2`. Used to enforce initial conditions.
    *(Note: Full integration of BC/IC losses into the main training loop's total loss is a key next step for solving well-posed problems.)*

## Prerequisites

-   **C Compiler**: GCC (or any C compiler supporting C99/C11).
-   **Make Utility**: For easy compilation using the provided `Makefile`.
-   **Operating System**: Developed on a Unix-like system (Linux/macOS). Uses `unistd.h` for `access()`, which might require MinGW or WSL on Windows.

## Installation and Compilation

1.  **Clone/Download**: Get the project files and place them in a directory, e.g., `arnab-mkj-pinn_c`.
2.  **Navigate to Project Directory**:
    ```bash
    cd path/to/arnab-mkj-pinn_c
    ```
3.  **Compile**:
    ```bash
    make
    ```
    This will generate three executables in the project root:
    -   `pinn`: The main PINN training program.
    -   `test_loss_functions`: Tests for PDE residual calculations.
    -   `test_neural_network`: Tests for neural network initialization and forward pass.

4.  **Clean Build Artifacts** (optional):
    ```bash
    make clean
    ```

## How to Run and Workflow

### 1. Running the PINN Training (`./pinn`)

The main program `pinn` trains the neural network.

**Syntax**:
```bash
./pinn --loss <loss_type> --activation <activation_function> [options...]
```
**Key Arguments**:

-   `--loss <type>`: Specify the PDE to solve.
    -   Available types: `schrodinger`, `maxwell`, `heat`, `wave`.
    -   This determines which PDE residual function will be used to calculate the loss.
-   `--activation <function>`: Activation function for the hidden layer of the neural network.
    -   Available functions: `relu`, `sigmoid`, `tanh`, `leaky_relu`.
    -   The choice of activation function can significantly impact the network's learning capability and the smoothness of the approximated solution.
-   `--epochs <N>`: Number of training iterations.
    -   Example: `--epochs 10000`.
    -   Each epoch involves processing `num_colloc` collocation points. More epochs generally lead to better convergence, but also take more time.
-   `--lr <rate>`: Initial learning rate for the stochastic gradient descent (SGD) optimizer.
    -   Example: `--lr 0.001`.
    -   The learning rate controls the step size when updating network weights. An adaptive learning rate decay is implemented, reducing the rate over epochs.
-   `--num_colloc <N_c>`: Number of collocation points sampled randomly from the domain per epoch.
    -   Example: `--num_colloc 100`.
    -   These points are used to evaluate the PDE residual and compute the loss. More points provide a better estimate of the true residual over the domain but increase computation per epoch.
-   `--dx <val>`: Spatial step size `Δx` used for approximating spatial derivatives via finite differences.
    -   Example: `--dx 0.01`.
    -   A smaller `dx` can lead to more accurate derivative approximations but may require a more capable network or finer features in the solution.
-   `--dt <val>`: Temporal step size `Δt` used for approximating time derivatives via finite differences.
    -   Example: `--dt 0.01`.
    -   Similar to `dx`, a smaller `dt` can improve accuracy but has computational implications. The choice of `dx` and `dt` can also affect the stability of the numerical scheme implicitly defined by the finite differences.

**PDE-Specific Physical Parameters**:
These arguments allow you to set the physical constants or coefficients for the chosen PDE.

-   `--potential <value>`: (For `--loss schrodinger`)
    -   Sets the potential `V` in the simplified Schrödinger equation.
    -   Example: `--potential 0.5`.
-   `--charge_density <value>`: (For `--loss maxwell`)
    -   Sets the charge density `ρ` (though not directly used in the current simplified 1D Maxwell residual, it's parsed for future extension).
    -   Example: `--charge_density 0.0`.
-   `--current_density <value>`: (For `--loss maxwell`)
    -   Sets the current density `J_z` in the 1D Maxwell's equations (Ampère's law term).
    -   Example: `--current_density 0.1`.
-   `--thermal_conductivity <value>`: (For `--loss heat`)
    -   Sets the thermal conductivity `k` for the heat equation. The thermal diffusivity `α` is then calculated as `k / DEFAULT_RHO_CP`.
    -   Example: `--thermal_conductivity 0.05`.
-   `--wave_speed <value>`: (For `--loss wave`)
    -   Sets the wave propagation speed `c` for the wave equation.
    -   Example: `--wave_speed 1.0`.

**Example Command**:
These commands assume you are in the root directory of the project where the `pinn` executable is located.

#### 1. Heat Equation
To train the PINN for the 1D Heat Equation using the `tanh` activation function for 20,000 epochs, with a learning rate of 0.001, thermal conductivity of 0.05, `dx=0.02`, `dt=0.005`, and 200 collocation points per epoch:

To train the PINN for the 1D Heat Equation:
$$ \frac{\partial u}{\partial t} = \alpha \frac{\partial^2 u}{\partial x^2} $$

```bash
./pinn --loss heat \
       --activation tanh \
       --epochs 15000 \
       --lr 0.001 \
       --num_colloc 150 \
       --dx 0.02 \
       --dt 0.005 \
       --thermal_conductivity 0.05 
```
       
--dx 0.02: Sets the spatial step for finite differences. \
--dt 0.005: Sets the time step for finite differences. \
--thermal_conductivity 0.05: Sets the k value (thermal diffusivity α will be derived from this). 

#### 2. Wave Equation
To train the PINN for the 1D Wave Equation:
$ \frac{\partial^2 u}{\partial t^2} = c^2 \frac{\partial^2 u}{\partial x^2} $

```bash
./pinn --loss wave \
       --activation relu \
       --epochs 20000 \
       --lr 0.0005 \
       --num_colloc 200 \
       --dx 0.025 \
       --dt 0.01 \
       --wave_speed 1.2
```

--dx 0.025: Sets the spatial step. \
--dt 0.01: Sets the time step. \
--wave_speed 1.2: Sets the wave propagation speed c. 

#### 3. Schrodinger Equation
To train the PINN for the simplified, real-valued form related to the Schrödinger Equation:
Residual form depends on the specific interpretation, e.g.,  $ R = ∂ψ/∂t + (1/ħ) * ((-ħ²/2m)∂²ψ/∂x² + Vψ) $

```bash
./pinn --loss schrodinger \
       --activation sigmoid \
       --epochs 25000 \
       --lr 0.001 \
       --num_colloc 100 \
       --dx 0.01 \
       --dt 0.001 \
       --potential 0.5
```

--dx 0.01: Sets the spatial step.\
--dt 0.001: Sets the time step (often needs to be small for time-dependent quantum systems).\
--potential 0.5: Sets the potential V(x) (assumed constant here).\


#### 4. Maxwell's Equation
To train the PINN for the simplified 1D Maxwell's Equations (e.g., TEz mode with Ez, Hy): \
$ R_1 = \frac{\partial E_z}{\partial x} + \mu_0 \frac{\partial H_y}{\partial t} $ \
$ R_2 = \frac{\partial H_y}{\partial x} + \epsilon_0 \frac{\partial E_z}{\partial t} + J_z $ \
(The network output NN_output[0] is treated as Ez, NN_output[1] as Hy. OUTPUT_SIZE in neural_network.h must be 2 for this).

```bash
./pinn --loss maxwell \
       --activation tanh \
       --epochs 30000 \
       --lr 0.0008 \
       --num_colloc 250 \
       --dx 0.015 \
       --dt 0.005 \
       --current_density 0.1 \
       --charge_density 0.0
```

--dx 0.015: Sets the spatial step.\
--dt 0.005: Sets the time step.\
--current_density 0.1: Sets the current density J_z.\
--charge_density 0.0: Sets the charge density ρ (note: charge_density is parsed but not actively used in the current maxwell_equations_residual for Gauss's law for E-field, which would be another equation to add for a fuller system).
   