from mpi4py import MPI
import numpy as np
import matplotlib.pyplot as plt

# Initialize the MPI environment
comm = MPI.COMM_WORLD
rank = comm.Get_rank()  # Get the rank of the process
size = comm.Get_size()  # Get the total number of processes
#Question 1
# def vanilla_gradient_descent(initial_x, learning_rate, num_iterations):
#     x = initial_x
#     for i in range(num_iterations):
#         gradient = 2 * x  # ∇f(x) = 2x
#         x = x - learning_rate * gradient
#         if rank == 0:  # Only the master process logs the progress
#             print(f"Iteration {i+1}: x = {x}, gradient = {gradient}")
#     return x

# # Example usage
# if rank == 0:  # Only the master process runs the gradient descent
#     initial_x = 3.5
#     learning_rate = 0.1
#     num_iterations = 20
#     result = vanilla_gradient_descent(initial_x, learning_rate, num_iterations)
#   

#     # Plot the function f(x) = x^2
#     x_vals = np.linspace(-4, 4, 500)
#     y_vals = x_vals**2

#     plt.figure(figsize=(8, 6))
#     plt.plot(x_vals, y_vals, label="f(x) = x^2", color="blue")

#     # Overlay the points visited by x during gradient descent
#     if rank == 0:  # Only the master process handles plotting
#         x_points = [initial_x]
#         x = initial_x
#         for _ in range(num_iterations):
#             gradient = 2 * x
#             x = x - learning_rate * gradient
#             x_points.append(x)
        
#         y_points = [xi**2 for xi in x_points]
#         plt.scatter(x_points, y_points, color="red", label="Gradient Descent Steps")
#         plt.plot(x_points, y_points, linestyle="--", color="red", alpha=0.6)

#         plt.title("Gradient Descent on f(x) = x^2")
#         plt.xlabel("x")
#         plt.ylabel("f(x)")
#         plt.axhline(0, color="black", linewidth=0.5, linestyle="--")
#         plt.axvline(0, color="black", linewidth=0.5, linestyle="--")
#         plt.legend()
#         plt.grid(True)
#         plt.show()

#     print(f"Final result: x = {result}")
# Question 2: Linear Regression with Gradient Descent
def y_hat(x, theta1, theta2):
    """Linear model: ŷ = θ1 * x + θ2"""
    return theta1 * x + theta2

def loss(y_true, y_pred):
    """Mean squared error"""
    return 0.5 * np.mean((y_true - y_pred)**2)

def gradient_theta1(x, y_true, y_pred):
    """Gradient of loss w.r.t. θ1"""
    return np.mean((y_pred - y_true) * x)

def gradient_theta2(x, y_true, y_pred):
    """Gradient of loss w.r.t. θ2"""
    return np.mean(y_pred - y_true)

# # ---------- Data Generation ----------
# np.random.seed(42)
# X = np.arange(0, 1, 0.01)
# Y = X + np.random.normal(0, 0.2, len(X))

# # ---------- Initialization ----------
# theta1 = -0.5
# theta2 = 0.2
# eta = 0.006
# E_max = 20
# history = []

# # ---------- Training Loop ----------
# for epoch in range(E_max):
#     y_pred = y_hat(X, theta1, theta2)
    
#     grad1 = gradient_theta1(X, Y, y_pred)
#     grad2 = gradient_theta2(X, Y, y_pred)
    
#     # Update parameters
#     theta1 -= eta * grad1
#     theta2 -= eta * grad2
    
#     current_loss = loss(Y, y_pred)
#     history.append((theta1, theta2, current_loss))
    
#     print(f"Epoch {epoch+1}/{E_max}: theta1={theta1:.4f}, theta2={theta2:.4f}, Loss={current_loss:.4f}")

# # ---------- Visualization ----------
# plt.figure(figsize=(12, 6))

# # Plot 1: Data and fits
# plt.subplot(1, 2, 1)
# plt.scatter(X, Y, label='Noisy Data', alpha=0.6)
# plt.plot(X, y_hat(X, 1.0, 0.0), 'r--', label='Optimal Fit (1.0, 0.0)')
# plt.plot(X, y_hat(X, theta1, theta2), 'g-', label='Learned Fit')
# plt.xlabel("X")
# plt.ylabel("Y")
# plt.title("Linear Regression Fit")
# plt.legend()

# # Plot 2: Convergence of parameters
# theta1_hist = [h[0] for h in history]
# theta2_hist = [h[1] for h in history]
# plt.subplot(1, 2, 2)
# plt.plot(theta1_hist, theta2_hist, 'bo-', label='Training Path')
# plt.plot(1.0, 0.0, 'r*', markersize=10, label='Optimal (1.0, 0.0)')
# plt.xlabel("theta1")
# plt.ylabel("theta2")
# plt.title("Parameter Convergence")
# plt.legend()
# plt.grid(True)

# plt.tight_layout()
# plt.savefig("non_parallel_regression_results.png", dpi=300)
# plt.show()

# # ---------- Final Output ----------
# print("\nFinal Results:")
# print(f"Learned parameters: theta1={theta1:.4f}, theta2={theta2:.4f}")
# print(f"Optimal parameters: theta1=1.0000, theta2=0.0000")
# print(f"Final loss: {history[-1][2]:.4f}")

# Question 3: Parallel Gradient Descent


def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Only the root process generates the full dataset
    if rank == 0:
        np.random.seed(42)
        X = np.arange(0, 1, 0.01)
        Y = X + np.random.normal(0, 0.2, len(X))
    else:
        X = None
        Y = None

    # Broadcast full data length and scatter chunks
    chunk_size = None
    if rank == 0:
        chunk_size = len(X) // size

    chunk_size = comm.bcast(chunk_size, root=0)
    
    x_chunk = np.empty(chunk_size, dtype='d')
    y_chunk = np.empty(chunk_size, dtype='d')
    
    comm.Scatter(X, x_chunk, root=0)
    comm.Scatter(Y, y_chunk, root=0)

    # Initialize parameters (shared across all ranks)
    theta1, theta2 = -0.5, 0.2
    eta = 0.006
    E_max = 20

    history = []
    for epoch in range(E_max):
        y_pred_chunk = y_hat(x_chunk, theta1, theta2)
        
        grad1_local = gradient_theta1(x_chunk, y_chunk, y_pred_chunk)
        grad2_local = gradient_theta2(x_chunk, y_chunk, y_pred_chunk)

        # Reduce gradients (sum and average across ranks)
        grad1_global = comm.allreduce(grad1_local, op=MPI.SUM) / size
        grad2_global = comm.allreduce(grad2_local, op=MPI.SUM) / size

        # Update shared parameters
        theta1 -= eta * grad1_global
        theta2 -= eta * grad2_global

        if rank == 0:
            full_y_pred = y_hat(X, theta1, theta2)
            current_loss = loss(Y, full_y_pred)
            history.append((theta1, theta2, current_loss))
            print(f"Epoch {epoch+1}/{E_max}: theta1={theta1:.4f}, theta2={theta2:.4f}, Loss={current_loss:.4f}")

    # Only rank 0 visualizes results
    if rank == 0:
        plt.figure(figsize=(12, 6))
        
        # Plot 1: Data and regression lines
        plt.subplot(1, 2, 1)
        plt.scatter(X, Y, label='Data', alpha=0.6)
        plt.plot(X, y_hat(X, 1.0, 0.0), 'r--', label='Optimal')
        plt.plot(X, y_hat(X, theta1, theta2), 'g-', label='Learned')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('Regression Fit')
        plt.legend()
        
        # Plot 2: Parameter convergence
        theta1_hist = [h[0] for h in history]
        theta2_hist = [h[1] for h in history]
        plt.subplot(1, 2, 2)
        plt.plot(theta1_hist, theta2_hist, 'b-o', markersize=4)
        plt.plot(1.0, 0.0, 'r*', markersize=10, label='Optimal')
        plt.xlabel('theta1')
        plt.ylabel('theta2')
        plt.title('Parameter Convergence')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig("distributed_regression_results.png", dpi=150)
        plt.show(block=True)

    MPI.Finalize()

if __name__ == "__main__":
    main()

