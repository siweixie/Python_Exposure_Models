import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

# Define the models in Python
def model_104(G, Q, Q_L, epsilon_L, gamma):
    C_avg = (gamma * G * (1 - epsilon_L)) / (Q + Q_L)
    C_L_avg = C_avg + (epsilon_L * gamma * G) / Q_L
    return C_avg, C_L_avg

def model_105(G, Q, Q_L, epsilon_L, V, t_g, T):
    time_vector = np.arange(0, T+1)
    C_rise = [(G * (1 - epsilon_L) / (Q + Q_L)) * (1 - np.exp(-(Q + Q_L) * t / V)) if t <= t_g else np.nan for t in time_vector]
    C0 = (G * (1 - epsilon_L)) / (Q + Q_L) * (1 - np.exp(-(Q + Q_L) * t_g / V))
    C_decay = [C0 * np.exp(-(Q + Q_L) * (t - t_g) / V) if t > t_g else np.nan for t in time_vector]
    return time_vector, C_rise, C_decay

def model_106(G, Q, Q_L, epsilon_L, Q_R, epsilon_RF, gamma):
    C_avg = (gamma * G * (1 - epsilon_L)) / ((Q + epsilon_RF * Q_R) + Q_L)
    C_L_avg = C_avg + (epsilon_L * gamma * G) / Q_L
    C_RF = C_avg * (1 - epsilon_RF)
    return C_avg, C_L_avg, C_RF

def model_107(G, Q, Q_L, epsilon_L, Q_R, epsilon_RF, V, t_g, T):
    Q_instead = Q + epsilon_RF * Q_R
    return model_105(G, Q_instead, Q_L, epsilon_L, V, t_g, T)

# Given Constants
T = 60
t_g = 15
G = 100
Q = 20
Q_L = 5
epsilon_L = 0.5
epsilon_L_F = 0.75
Q_R = 5
epsilon_RF = 0.9
V = 100
gamma = 0.25

# Calculate Results
results_104 = model_104(G, Q, Q_L, epsilon_L, gamma)
results_105 = model_105(G, Q, Q_L, epsilon_L, V, t_g, T)
results_106 = model_106(G, Q, Q_L, epsilon_L, Q_R, epsilon_RF, gamma)
results_107 = model_107(G, Q, Q_L, epsilon_L, Q_R, epsilon_RF, V, t_g, T)

# Prepare Data for Plotting and Comparison
time_105, C_rise_105, C_decay_105 = results_105
time_107, C_rise_107, C_decay_107 = results_107

results_105_df = pd.DataFrame({
    'Time': np.concatenate([time_105, time_105]),
    'Concentration': np.concatenate([C_rise_105, C_decay_105]),
    'Model': 'Model 105'
}).dropna()

results_107_df = pd.DataFrame({
    'Time': np.concatenate([time_107, time_107]),
    'Concentration': np.concatenate([C_rise_107, C_decay_107]),
    'Model': 'Model 107'
}).dropna()

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(results_105_df['Time'], results_105_df['Concentration'], label='Model 105')
plt.plot(results_107_df['Time'], results_107_df['Concentration'], label='Model 107', linestyle='dashed')
plt.title('Comparison of Model 105 and Model 107')
plt.xlabel('Time (minutes)')
plt.ylabel('Concentration (mg/m^3)')
plt.legend()
plt.grid(True)
plt.show()

# Calculate Mean Squared Error and R^2
mse = np.mean((results_105_df['Concentration'] - results_107_df['Concentration'])**2)
r2 = np.corrcoef(results_105_df['Concentration'], results_107_df['Concentration'])[0, 1]**2

# Normality Test
normality_test_105 = stats.shapiro(results_105_df['Concentration'])
normality_test_107 = stats.shapiro(results_107_df['Concentration'])

# Wilcoxon Signed-Rank Test
wilcoxon_test = stats.wilcoxon(results_105_df['Concentration'], results_107_df['Concentration'])

mse, r2, normality_test_105, normality_test_107, wilcoxon_test
