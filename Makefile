CC=gcc
CFLAGS=-Iinclude -Wall -Wextra

all: pinn test_loss_functions test_neural_network

pinn: src/main.c src/neural_network.c src/loss_functions.c src/utils.c
	$(CC) -o pinn src/main.c src/neural_network.c src/loss_functions.c src/utils.c $(CFLAGS) -lm

test_loss_functions: test_cases/test_loss_functions.c src/loss_functions.c
	$(CC) -o test_loss_functions test_cases/test_loss_functions.c src/loss_functions.c $(CFLAGS) -lm

test_neural_network: test_cases/test_neural_network.c src/neural_network.c
	$(CC) -o test_neural_network test_cases/test_neural_network.c src/loss_functions.c src/neural_network.c src/utils.c $(CFLAGS) -lm

clean:
	rm -f pinn test_loss_functions test_neural_network