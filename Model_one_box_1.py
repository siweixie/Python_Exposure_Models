import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import shapiro, wilcoxon

# Define the models in Python
def model_100(G, Q, gamma):
    C_steady = gamma * G / Q
    return C_steady

def model_101(G, Q, V, t_g, T):
    time_vector = np.arange(0, T + 1)
    C_rise = [(G / Q) * (1 - np.exp((-Q * t) / V)) if t <= t_g else np.nan for t in time_vector]
    C0 = (G / Q) * (1 - np.exp(-Q * t_g / V))
    C_decay = [C0 * np.exp(-Q * (t - t_g) / V) if t > t_g else np.nan for t in time_vector]
    C_combined = np.array([x if not np.isnan(x) else y for x, y in zip(C_rise, C_decay)])
    return time_vector, C_combined

def model_103(G, Q, Q_R, epsilon_RF, V, t_g, T):
    Q_with_RF = Q + epsilon_RF * Q_R
    return model_101(G, Q_with_RF, V, t_g, T)

# Set the parameters
T = 60   # Total time (minutes)
t_g = 15 # Time of generation (minutes)
G = 100  # mg/min
Q = 20   # m^3/min
Q_R = 5  # m^3/min
epsilon_RF = 0.9  # Efficiency of recirculation filtration
V = 100  # m^3
gamma = 0.25  

# Perform calculations
C_steady_100 = model_100(G, Q, gamma)
time_101, concentration_101 = model_101(G, Q, V, t_g, T)
time_103, concentration_103 = model_103(G, Q, Q_R, epsilon_RF, V, t_g, T)

# Visualization: Model 101 vs Model 103
plt.figure(figsize=(10, 5))
plt.plot(time_101, concentration_101, label='Model 101')
plt.plot(time_103, concentration_103, label='Model 103', linestyle='--')
plt.title('Model 101 vs Model 103')
plt.xlabel('Time (minutes)')
plt.ylabel('Concentration (mg/m^3)')
plt.legend()
plt.show()

# Calculate Mean Squared Error and R^2
mse = np.mean((concentration_101 - concentration_103)**2)
r2 = np.corrcoef(concentration_101, concentration_103)[0, 1]**2

# Statistical Analysis
# Shapiro-Wilk test for normality
normality_test_101 = shapiro(concentration_101[~np.isnan(concentration_101)])
normality_test_103 = shapiro(concentration_103[~np.isnan(concentration_103)])

# Wilcoxon Signed-Rank Test (if distributions are not normal)
if normality_test_101.pvalue < 0.05 and normality_test_103.pvalue < 0.05:
    # Ensuring equal length for the test
    min_length = min(len(concentration_101[~np.isnan(concentration_101)]), 
                     len(concentration_103[~np.isnan(concentration_103)]))
    wilcoxon_test = wilcoxon(concentration_101[:min_length], 
                             concentration_103[:min_length])

mse, r2, normality_test_101, normality_test_103, wilcoxon_test
