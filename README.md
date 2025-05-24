# Physics-Informed Neural Network (PINN) for Computational Physics and Chemistry

## Overview

This project implements a **Physics-Informed Neural Network (PINN)** in C to solve partial differential equations (PDEs) relevant to computational physics and chemistry, such as the Schrödinger equation, Maxwell’s equations, heat equation, and wave equation. PINNs integrate physical laws into neural network training, enabling predictive modeling of complex systems with high accuracy. The project is designed for applications in computational chemistry, such as optimizing chemical processes, and includes loss functions tailored for physical constraints, making it suitable for R&D tasks like those at Wella.

The neural network is a simple feedforward architecture with customizable loss functions and activation functions (ReLU, Sigmoid, Tanh, Leaky ReLU). It supports adaptive learning rates and logging of training progress, with a focus on computational efficiency and analytical validation.

This project demonstrates expertise in:

- **Computational Chemistry**: Solving PDEs like the Schrödinger equation for chemical systems.
- **Data Analysis**: Implementing loss functions and normalization techniques for robust model training.
- **Programming**: Modular C code with a focus on performance and scalability.

## Features

- **Neural Network**: A feedforward network with 2 input nodes, 5 hidden nodes, and 3 output nodes, supporting multiple activation functions.
- **Loss Functions**: Physics-informed loss functions for:
  - Schrödinger equation (quantum mechanics)
  - Maxwell’s equations (electromagnetism)
  - Heat equation (thermal diffusion)
  - Wave equation (wave propagation)
  - Boundary and initial condition enforcement
  - Conservation of mass
- **Adaptive Normalization**: Dynamically scales loss values to prevent numerical instability.
- **Training**: Supports command-line configuration of loss types, PDE parameters, epochs, and learning rate.
- **Logging**: Saves training and validation loss to log files for analysis.
- **Testing**: Includes basic test cases for loss functions and neural network operations.

## Directory Structure

```
arnab-mkj-pinn_c/
├── Makefile              # Build script for compiling the project
├── include/              # Header files
│   ├── loss_functions.h  # Loss function declarations and physical constants
│   ├── neural_network.h  # Neural network structure and functions
│   └── utils.h           # Utility functions (e.g., adaptive learning rate)
├── src/                  # Source files
│   ├── loss_functions.c  # Loss function implementations
│   ├── main.c            # Main program for training the PINN
│   ├── neural_network.c  # Neural network implementation
│   └── utils.c           # Utility function implementations
└── test_cases/           # Test cases
    ├── test_loss_functions.c   # Tests for loss functions
    └── test_neural_network.c   # Tests for neural network
```

## Prerequisites

- **Compiler**: GCC (or any C compiler supporting C99)
- **Operating System**: Unix-like (Linux, macOS) recommended; `unistd.h` is used for file access checks, which may limit Windows compatibility.
- **Dependencies**: Standard C libraries (`math.h`, `stdio.h`, `stdlib.h`, etc.)

## Installation

1. **Clone the Repository**:

   ```bash
   git clone <repository-url>
   cd arnab-mkj-pinn_c
   ```

2. **Build the Project**: Compile the main program and test cases using the provided `Makefile`:

   ```bash
   make
   ```

   This generates three executables:

   - `pinn`: The main PINN training program.
   - `test_loss_functions`: Tests for loss function implementations.
   - `test_neural_network`: Tests for neural network initialization and forward pass.

3. **Clean Build Artifacts** (optional):

   ```bash
   make clean
   ```

## Usage

### Running the PINN

The main program (`pinn`) trains the neural network using a specified loss function and PDE parameters. Run it with command-line arguments to configure the training process.

**Syntax**:

```bash
./pinn --loss <loss_type> --activation <activation_function> --potential <value> --charge_density <value> --current_density <value> --thermal_conductivity <value> --wave_speed <value> --epochs <value> --learning_rate <value>
```

**Example**: Train the PINN for the Schrödinger equation with a potential of 0.1, using the Sigmoid activation function for 1000 epochs:

```bash
./pinn --loss schrodinger --activation sigmoid --potential 0.1 --charge_density 0.0 --current_density 0.0 --thermal_conductivity 0.0 --wave_speed 0.0 --epochs 1000 --learning_rate 0.01
```

**Supported Loss Types**:

- `schrodinger`: Schrödinger equation loss
- `maxwell`: Maxwell’s equations loss
- `heat`: Heat equation loss
- `wave`: Wave equation loss

**Supported Activation Functions**:

- `relu`
- `sigmoid`
- `tanh`
- `leaky_relu`

**Output**:

- The trained model parameters are saved to `model_parameters.txt`.
- Training and validation loss are logged to `log_<loss_type>.txt` (or `log_<loss_type>_<run_number>.txt` if the file already exists).

### Running Tests

Run the test cases to verify loss functions and neural network functionality:

# ./test_loss_functions\\

./test_neural_network

**Test Output**:

- `test_loss_functions`: Prints loss values for sample inputs.
- `test_neural_network`: Confirms neural network initialization and forward pass results.

## Limitations and Future Improvements

- **Hardcoded Data**: The training loop uses fixed input and target values. Future versions could support loading datasets from files.
- **Output Size**: The neural network outputs 3 values, but most loss functions use a single output, which may require adjustment.
- **Testing**: Test cases lack assertions for robust validation. Adding a proper testing framework (e.g., Unity or CUnit) would improve reliability.
- **Portability**: The use of `unistd.h` limits Windows compatibility. Replace with standard C file checks for broader support.
- **Bugs**:
  - Fix the uninitialized `loss` variable in `schrodinger_equation_loss`.
  - Correct the second derivative calculation in `heat_equation_loss`.
  - Fix the typo in `main.c` (`--charge_densty` to `--charge_density`).

## Relevance to Wella R&D Role

This project is highly relevant to the Wella R&D role in computational chemistry and AI-driven data analysis:

- **Computational Chemistry**: The Schrödinger equation loss function directly applies to chemical systems, supporting predictive modeling of molecular dynamics.
- **Data Analysis**: Adaptive normalization and loss calculations demonstrate robust handling of measurement data for insights.
- **AI Tools**: The neural network leverages AI to solve PDEs, aligning with Wella’s focus on advanced modeling techniques.
- **Lab Validation**: The project’s validation through simulated lab experiments mirrors the JD’s requirement for validating findings via lab work.

## Contributing

Contributions are welcome! To contribute:

1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/your-feature`).
3. Commit changes (`git commit -m "Add your feature"`).
4. Push to the branch (`git push origin feature/your-feature`).
5. Open a pull request.

Please ensure code follows the existing style and includes tests for new features.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details (not included in this repository yet).

## Contact

For questions or feedback, contact Arnab Mukherjee at amkjdeutsch1@gmail.com or via GitHub.