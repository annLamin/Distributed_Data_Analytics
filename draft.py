from mpi4py import MPI
import numpy as np
import matplotlib.pyplot as plt

def y_hat(x, theta1, theta2):
    return theta1 * x + theta2

def loss(y_true, y_pred):
    return 0.5 * np.mean((y_true - y_pred)**2)

def gradient_theta1(x, y_true, y_pred):
    return np.mean((y_pred - y_true) * x)

def gradient_theta2(x, y_true, y_pred):
    return np.mean(y_pred - y_true)

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    
    if rank == 0:
        # Generate and prepare data
        np.random.seed(42)
        X = np.arange(0, 1, 0.01)
        Y = X + np.random.normal(0, 0.2, len(X))
        
        # Initialize parameters
        theta1, theta2 = -0.5, 0.2
        eta = 0.006
        E_max = 20
        history = []

        # Training loop
        for epoch in range(E_max):
            y_pred = y_hat(X, theta1, theta2)
            grad1 = gradient_theta1(X, Y, y_pred)
            grad2 = gradient_theta2(X, Y, y_pred)
            
            theta1 -= eta * grad1
            theta2 -= eta * grad2
            
            current_loss = loss(Y, y_pred)
            history.append((theta1, theta2, current_loss))
            print(f"Epoch {epoch+1}/{E_max}: theta1={theta1:.4f}, theta2={theta2:.4f}, Loss={current_loss:.4f}")

        # Visualization
        plt.figure(figsize=(12, 6))
        
        # Plot 1: Data and regression lines
        plt.subplot(1, 2, 1)
        plt.scatter(X, Y, label='Data', alpha=0.6)
        plt.plot(X, y_hat(X, 1.0, 0.0), 'r--',markersize=10, label='Optimal (1.0, 0.0)')
        plt.plot(X, y_hat(X, theta1, theta2), 'g-', label='Learned')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('Regression Fit')
        plt.legend()
        
        # Plot 2: Parameter convergence
        plt.subplot(1, 2, 2)
        theta1_hist = [h[0] for h in history]
        theta2_hist = [h[1] for h in history]
        plt.plot(theta1_hist, theta2_hist, 'b-o', markersize=4)
        plt.plot(1.0, 0.0, 'r*', markersize=10, label='Optimal')
        plt.xlabel('theta1')
        plt.ylabel('theta2')
        plt.title('Parameter Convergence')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('regression_results.png', dpi=150)
        plt.show(block=True)  # block=True keeps the plot window open
        
    MPI.Finalize()

if __name__ == "__main__":
    main()