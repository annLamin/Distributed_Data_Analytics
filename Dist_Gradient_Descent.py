from mpi4py import MPI
import numpy as np

# Initialize the MPI environment
comm = MPI.COMM_WORLD
rank = comm.Get_rank()  # Get the rank of the process
size = comm.Get_size()  # Get the total number of processes
#Question 1
def vanilla_gradient_descent(initial_x, learning_rate, num_iterations):
    x = initial_x
    for i in range(num_iterations):
        gradient = 2 * x  # ∇f(x) = 2x
        x = x - learning_rate * gradient
        if rank == 0:  # Only the master process logs the progress
            print(f"Iteration {i+1}: x = {x}, gradient = {gradient}")
    return x

# Example usage
if rank == 0:  # Only the master process runs the gradient descent
    initial_x = 3.5
    learning_rate = 0.1
    num_iterations = 20
    result = vanilla_gradient_descent(initial_x, learning_rate, num_iterations)
    import matplotlib.pyplot as plt

    # Plot the function f(x) = x^2
    x_vals = np.linspace(-4, 4, 500)
    y_vals = x_vals**2

    plt.figure(figsize=(8, 6))
    plt.plot(x_vals, y_vals, label="f(x) = x^2", color="blue")

    # Overlay the points visited by x during gradient descent
    if rank == 0:  # Only the master process handles plotting
        x_points = [initial_x]
        x = initial_x
        for _ in range(num_iterations):
            gradient = 2 * x
            x = x - learning_rate * gradient
            x_points.append(x)
        
        y_points = [xi**2 for xi in x_points]
        plt.scatter(x_points, y_points, color="red", label="Gradient Descent Steps")
        plt.plot(x_points, y_points, linestyle="--", color="red", alpha=0.6)

        plt.title("Gradient Descent on f(x) = x^2")
        plt.xlabel("x")
        plt.ylabel("f(x)")
        plt.axhline(0, color="black", linewidth=0.5, linestyle="--")
        plt.axvline(0, color="black", linewidth=0.5, linestyle="--")
        plt.legend()
        plt.grid(True)
        plt.show()

# Question 2
