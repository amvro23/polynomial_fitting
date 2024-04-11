import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def polynomial_fit(x, *coefficients):
    return sum(coefficients[i] * x**i for i in range(len(coefficients)))

def predict(x, y, degree):
    if len(x) >= (degree + 1):
        initial_guess = np.ones(degree + 1)  # Generate initial guess for coefficients
        popt, pcov = curve_fit(polynomial_fit, x, y, p0=initial_guess)
        x_test = np.linspace(x[0], x[-1], 250)
        y_pred = polynomial_fit(x, *popt)
        
        plt.plot(x_test, polynomial_fit(x_test, *popt), 'k--', label='predicted data')
        plt.plot(x, y, 'mo', mfc='None', label='observed data')
        plt.legend()
        plt.show()

        # Print coefficients
        for i, coef in enumerate(popt):
            print(f'a{i} = {coef}')

        # Calculate R-squared
        residuals = y - y_pred
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((y - np.mean(y))**2)
        r_squared = 1 - (ss_res / ss_tot)

        print('The R2 values was calculated to be', r_squared)
        
        return popt, r_squared
    else:
        print('Please provide more x-values')


x = np.array([0.0, 1.0, 2.0, 3.0,  4.0,  5.0])
y = np.array([0.0, 0.8, 0.9, 0.1, -0.8, -1.0])

print(predict(x, y, 3))