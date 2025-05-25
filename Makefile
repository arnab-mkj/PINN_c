CC=gcc
# Add -std=c99 or -std=c11 if using features from those standards explicitly
# Add -g for debugging symbols
CFLAGS=-Iinclude -Wall -Wextra -g -std=c99 
LDFLAGS=-lm # Link math library

# Executables
TARGETS=pinn test_loss_functions test_neural_network

# Source files
SRC_MAIN=src/main.c src/neural_network.c src/loss_functions.c src/utils.c
SRC_TEST_LOSS=test_cases/test_loss_functions.c src/loss_functions.c src/neural_network.c src/utils.c # test_loss now needs NN for context
SRC_TEST_NN=test_cases/test_neural_network.c src/neural_network.c src/utils.c src/loss_functions.c # Added loss_functions for ActivationFunction enum if used

all: $(TARGETS)

pinn: $(SRC_MAIN)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

test_loss_functions: $(SRC_TEST_LOSS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

test_neural_network: $(SRC_TEST_NN)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

clean:
	rm -f $(TARGETS) *.o log_*.txt pinn_model_trained.txt

.PHONY: all clean