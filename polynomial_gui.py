import tkinter as tk
from tkinter import ttk
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

class PolynomialFitApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Polynomial Fitting")

        # Frame to hold inputs
        input_frame = ttk.Frame(root, padding="40")
        input_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # X values entry
        ttk.Label(input_frame, text="X values (comma-separated):").grid(row=0, column=0, sticky=tk.W)
        self.x_entry = ttk.Entry(input_frame, width=50)
        self.x_entry.grid(row=0, column=1)

        # Y values entry
        ttk.Label(input_frame, text="Y values (comma-separated):").grid(row=1, column=0, sticky=tk.W)
        self.y_entry = ttk.Entry(input_frame, width=50)
        self.y_entry.grid(row=1, column=1)

        # Degree of polynomial entry
        ttk.Label(input_frame, text="Degree of polynomial:").grid(row=2, column=0, sticky=tk.W)
        self.degree_entry = ttk.Entry(input_frame, width=10)
        self.degree_entry.grid(row=2, column=1)

        # Result label
        self.result_label = ttk.Label(root, text="")
        self.result_label.grid(row=1, column=0, pady=(10, 0))

        # Fit button
        fit_button = ttk.Button(root, text="Fit Polynomial", command=self.fit_polynomial)
        fit_button.grid(row=2, column=0, pady=(10, 20))

    def polynomial_fit(self, x, *coefficients):
        """
        Polynomial fitting equation
        """
        return sum(coefficients[i] * x**i for i in range(len(coefficients)))

    def fit_polynomial(self):
        x_str = self.x_entry.get().strip()
        y_str = self.y_entry.get().strip()
        degree_str = self.degree_entry.get().strip()

        try:
            x_values = np.array([float(val) for val in x_str.split(',')])
            y_values = np.array([float(val) for val in y_str.split(',')])
            degree = int(degree_str)

            if len(x_values) != len(y_values):
                raise ValueError("Number of X values must be equal to number of Y values")

            if len(x_values) < (degree + 1):
                raise ValueError("Please provide more X values")

            initial_guess = np.ones(degree + 1)  # Generate initial guess for coefficients
            popt, pcov = curve_fit(self.polynomial_fit, x_values, y_values, p0=initial_guess)

            # Calculate R-squared
            y_pred = self.polynomial_fit(x_values, *popt)
            ss_res = np.sum((y_values - y_pred)**2)
            ss_tot = np.sum((y_values - np.mean(y_values))**2)
            r_squared = 1 - (ss_res / ss_tot)

            # Display optimized coefficients and R-squared
            coefficients_text = "\n".join([f"a{i} = {coef}" for i, coef in enumerate(popt)])
            result_text = f"Optimized Coefficients:\n{coefficients_text}\nR2 = {r_squared:.4f}"

            # Plot the fit
            x_test = np.linspace(x_values.min(), x_values.max(), 250)
            plt.plot(x_test, self.polynomial_fit(x_test, *popt), 'k--', label='predicted data')
            plt.plot(x_values, y_values, 'mo', mfc='None', label='observed data')
            plt.legend()
            plt.show()

            self.result_label.config(text=result_text)

        except Exception as e:
            self.result_label.config(text=f"Error: {e}")

def main():
    root = tk.Tk()
    app = PolynomialFitApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
