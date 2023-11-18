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
    return time_vector, C_rise, C_decay

def model_102(G, Q, Q_R, epsilon_RF, gamma):
    C_bar = (gamma * G) / (Q + epsilon_RF * Q_R)
    return C_bar

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


# Calculate Results
results_100 = model_100(G, Q, gamma)
results_101 = model_101(G, Q, V, t_g, T)
results_102 = model_102(G, Q, Q_R, epsilon_RF, gamma)
results_103 = model_103(G, Q, Q_R, epsilon_RF, V, t_g, T)


# Prepare Data for Plotting and Comparison
time_101, C_rise_101, C_decay_101 = results_101
time_103, C_rise_103, C_decay_103 = results_103

results_101_df = pd.DataFrame({
    'Time': np.concatenate([time_101, time_101]),
    'Concentration': np.concatenate([C_rise_101, C_decay_101]),
    'Model': 'Model 101'
}).dropna()

results_103_df = pd.DataFrame({
    'Time': np.concatenate([time_103, time_103]),
    'Concentration': np.concatenate([C_rise_103, C_decay_103]),
    'Model': 'Model 103'
}).dropna()

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(results_101_df['Time'], results_101_df['Concentration'], label='Model 101')
plt.plot(results_103_df['Time'], results_103_df['Concentration'], label='Model 103', linestyle='dashed')
plt.title('Comparison of Model 101 and Model 103')
plt.xlabel('Time (minutes)')
plt.ylabel('Concentration (mg/m^3)')
plt.legend()
plt.grid(True)
plt.show()

# Calculate Mean Squared Error and R^2
mse = np.mean((results_101_df['Concentration'] - results_103_df['Concentration'])**2)
r2 = np.corrcoef(results_101_df['Concentration'], results_103_df['Concentration'])[0, 1]**2

# Normality Test
normality_test_101 = stats.shapiro(results_101_df['Concentration'])
normality_test_103 = stats.shapiro(results_103_df['Concentration'])

# Wilcoxon Signed-Rank Test
wilcoxon_test = stats.wilcoxon(results_101_df['Concentration'], results_103_df['Concentration'])

mse, r2, normality_test_101, normality_test_103, wilcoxon_test
