import numpy as np
import matplotlib.pyplot as plt
import time

"""Creating a class to compare linear scan and gradient descent for finding which is better in finding the global minima (comaprison btw them)"""
class LinearRegressionComparison:
    """Using init to initialize the dataset with a given range and true parameters"""
    def __init__(self, x_range=(-10, 10), num_points=100, true_m1=2, true_m=5):
        self.x = np.linspace(x_range[0], x_range[1], num_points)
        self.true_m1 = true_m1
        self.true_m = true_m
        self.y = self.true_m1 * self.x + self.true_m + np.random.normal(0, 1, len(self.x))
        self.best_m1_linear = None
        self.best_m1_gd = None

    """Compute Mean Squared Error (MSE) for a given slope m1 and intercept m"""
    def mse(self, m1, m):
        y_pred = m1 * self.x + m
        return np.mean((self.y - y_pred) ** 2)
    
    """Perform linear search to find the best m1 by keeping m as constant"""
    def linear_scan(self, m1_range=(0, 4), steps=100):
        m1_values = np.linspace(m1_range[0], m1_range[1], steps)
        min_loss = float('inf')
        best_m1 = None
        
        start_time = time.time()
        for m1 in m1_values:
            loss = self.mse(m1, self.true_m)
            if loss < min_loss:
                min_loss = loss
                best_m1 = m1
        linear_scan_time = time.time() - start_time
        
        self.best_m1_linear = best_m1
        print(f"Linear Scan -> Best m1: {best_m1}, Time: {linear_scan_time:.4f} sec")

        """Plotting the obtained results from performing linear scan"""
        plt.figure(figsize=(8, 5))
        plt.scatter(self.x, self.y, color='blue', label='Original Data', alpha=0.5)
        y_pred_linear = self.best_m1_linear * self.x + self.true_m
        plt.plot(self.x, y_pred_linear, color='red', label=f'Linear Scan (m1={self.best_m1_linear:.2f})')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Linear Scan Fit')
        plt.legend()
        plt.grid(True)
        plt.show()


    """Performing gradient descent to find  the best m1"""
    def gradient_descent(self, learning_rate=0.01, iterations=100, initial_m1=0):
        m1 = initial_m1
        start_time = time.time()
        
        for _ in range(iterations):
            error = self.y - (m1 * self.x + self.true_m)
            gradient = (-2 / len(self.y)) * np.sum(self.x * error)
            m1 -= learning_rate * gradient
        
        gd_time = time.time() - start_time
        self.best_m1_gd = m1
        print(f"Gradient Descent -> Best m1: {m1}, Time: {gd_time:.4f} sec")
        
        """Plotting results for gradient descent"""
        plt.figure(figsize=(8, 5))
        plt.scatter(self.x, self.y, color='blue', label='Original Data', alpha=0.5)
        y_pred_gd = self.best_m1_gd * self.x + self.true_m
        plt.plot(self.x, y_pred_gd, color='green', linestyle='dashed', label=f'Gradient Descent (m1={self.best_m1_gd:.2f})')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Gradient Descent Fit')
        plt.legend()
        plt.grid(True)
        plt.show()

    """Finally comparing the both results (obatined from linear and gradient descent) to see which is optimal"""
    def plot_comparison(self):
        plt.figure(figsize=(10, 6))
        plt.scatter(self.x, self.y, color='blue', label='Original Data', alpha=0.5)
        y_pred_linear = self.best_m1_linear * self.x + self.true_m
        plt.plot(self.x, y_pred_linear, color='red', label=f'Linear Scan (m1={self.best_m1_linear:.2f})')
        y_pred_gd = self.best_m1_gd * self.x + self.true_m
        plt.plot(self.x, y_pred_gd, color='green', linestyle='dashed', label=f'Gradient Descent (m1={self.best_m1_gd:.2f})')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Comparison of Linear Scan and Gradient Descent')
        plt.legend()
        plt.grid(True)
        plt.show()
    
# calling the functions from the class
model = LinearRegressionComparison()
model.linear_scan()
model.gradient_descent()
model.plot_comparison()
