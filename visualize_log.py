import matplotlib.pyplot as plt
import numpy as np
import re # Keep re for filename parsing if needed, though simple string ops might be enough

def visualize_training_log(log_filename):
    epochs = []
    losses = []

    # Read log data from file
    try:
        with open(log_filename, 'r') as file:
            header = file.readline().strip() # Read and verify header
            if header != "Epoch,AverageLoss":
                print(f"Warning: Log file header is not 'Epoch,AverageLoss'. Found: '{header}'")
                # Attempt to proceed if format seems compatible, or exit
                # For now, we'll assume it's a critical mismatch if not exact.
                # return # Or raise an error

            for line in file:
                try:
                    parts = line.strip().split(',')
                    if len(parts) == 2:
                        epoch = int(parts[0])
                        loss = float(parts[1])
                        epochs.append(epoch)
                        losses.append(loss)
                    elif line.strip(): # Non-empty line that doesn't match
                        print(f"Skipping malformed line: {line.strip()}")
                except ValueError as e:
                    print(f"Error processing line: {line.strip()}. Error: {e}")
                    continue
    except FileNotFoundError:
        print(f"Error: Log file '{log_filename}' not found.")
        return
    except Exception as e:
        print(f"An error occurred while reading the file: {e}")
        return

    if not epochs or not losses:
        print("No data found in log file to plot.")
        return

    # Use logarithmic scale for the y-axis directly in plot for better handling of wide loss ranges
    # Or, if normalization is still desired for visual comparison between different runs on a 0-1 scale:
    def log_normalize_positive(values):
        if not values:
            return []

        # Ensure all values are positive for log
        # If losses can be zero or negative, this needs careful handling.
        # Our loss is squared residual, so it should be >= 0.
        # Adding a small epsilon to avoid log(0) if loss can be exactly 0.
        values_array = np.array(values) + 1e-12

        log_values = np.log(values_array)

        min_log_val = np.min(log_values)
        max_log_val = np.max(log_values)

        if max_log_val == min_log_val: # Avoid division by zero if all log values are the same
            return np.zeros_like(log_values)

        return (log_values - min_log_val) / (max_log_val - min_log_val)

    normalized_losses = log_normalize_positive(losses)

    # Plot training loss
    plt.figure(figsize=(12, 6))

    # Option 1: Plot normalized loss (0 to 1 scale)
    # plt.plot(epochs, normalized_losses, label='Log-Normalized Training Loss', color='blue', linestyle='-', marker='.', markersize=3)
    # plt.ylabel('Log-Normalized Training Loss', fontsize=14)
    # plt.ylim(0, 1.1) # Optional: fix y-axis for normalized view

    # Option 2: Plot actual loss on a log scale (often more informative for loss curves)
    plt.plot(epochs, losses, label='Training Loss', color='blue', linestyle='-', marker='.', markersize=3)
    plt.yscale('log')
    plt.ylabel('Training Loss (Log Scale)', fontsize=14)


    # --- Title Generation ---
    # Filename example: log_heat_tanh.txt or log_heat_tanh_1.txt
    base_filename = log_filename.replace("log_", "").replace(".txt", "") # e.g., heat_tanh or heat_tanh_1

    parts = base_filename.split('_')
    pde_type_key = parts[0] # e.g., "heat"

    pde_display_names = {
        "schrodinger": "Schrödinger Eq.",
        "maxwell": "Maxwell's Eqs.",
        "heat": "Heat Eq.",
        "wave": "Wave Eq."
    }
    pde_display_name = pde_display_names.get(pde_type_key, pde_type_key.capitalize()) # Default to capitalized key if not found

    # Construct title
    title_suffix = base_filename # e.g. heat_tanh or heat_tanh_1
    plot_title = f'{pde_display_name} Training Progress ({title_suffix})'


    # Labeling the plot
    plt.xlabel('Epoch', fontsize=14)
    plt.title(plot_title, fontsize=16)
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.5) # Grid for both major and minor ticks on log scale
    plt.tight_layout()

    # Save the plot to a file
    save_filename = f'plot_{base_filename}.png'
    plt.savefig(save_filename, dpi=300)
    print(f"Plot saved as {save_filename}")

    # Show the plot
    plt.show()

if __name__ == '__main__':
    # Example usage:
    # visualize_training_log('log_heat_tanh.txt')
    # visualize_training_log('log_wave_relu_1.txt')

    # You can make this script take a command-line argument for the filename
    import sys
    if len(sys.argv) > 1:
        log_file_to_visualize = sys.argv[1]
        visualize_training_log(log_file_to_visualize)
    else:
        print("Usage: python visualize_log.py <path_to_log_file.txt>")
        # As a fallback, try a default log name if no argument is given
        # print("Attempting to visualize 'log_heat_tanh.txt' as a default...")
        # visualize_training_log('log_heat_tanh.txt')