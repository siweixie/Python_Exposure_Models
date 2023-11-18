import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import shapiro, wilcoxon

# Define the models in Python
def model_108(G, Q, Q_L, epsilon_L, epsilon_L_F, gamma):
    C_avg = (gamma * G * (1 - epsilon_L * epsilon_L_F)) / (Q + epsilon_L_F * Q_L)
    C_LE = C_avg + (epsilon_L * gamma * G) / Q_L
    C_LF = C_LE * (1 - epsilon_L_F)
    return C_avg, C_LE, C_LF

def model_109(G, Q, Q_L, epsilon_L, epsilon_L_F, V, t_g, T):
    time_vector = np.arange(0, T + 1)
    C_rise = [(G * (1 - epsilon_L * epsilon_L_F) / (Q + epsilon_L_F * Q_L)) * (1 - np.exp(-(Q + epsilon_L_F * Q_L) * t / V)) if t <= t_g else np.nan for t in time_vector]
    C0 = (G * (1 - epsilon_L * epsilon_L_F)) / (Q + epsilon_L_F * Q_L) * (1 - np.exp(-(Q + epsilon_L_F * Q_L) * t_g / V))
    C_decay = [C0 * np.exp(-(Q + epsilon_L_F * Q_L) * (t - t_g) / V) if t > t_g else np.nan for t in time_vector]
    return time_vector, C_rise, C_decay

def model_110(G, Q, Q_L, epsilon_L, epsilon_L_F, Q_R, epsilon_RF, gamma):
    C_avg = (gamma * G * (1 - epsilon_L * epsilon_L_F)) / ((Q + epsilon_RF * Q_R) + epsilon_L_F * Q_L)
    C_LE = C_avg + (epsilon_L * gamma * G) / Q_L
    C_LF = C_LE * (1 - epsilon_L_F)
    C_RF = C_avg * (1 - epsilon_RF)
    return C_avg, C_LE, C_LF, C_RF

def model_111(G, Q, Q_L, epsilon_L, epsilon_L_F, Q_R, epsilon_RF, V, t_g, T):
    Q_instead = Q + epsilon_RF * Q_R
    return model_109(G, Q_instead, Q_L, epsilon_L, epsilon_L_F, V, t_g, T)


# Set the parameters
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
results_108 = model_108(G, Q, Q_L, epsilon_L, epsilon_L_F, gamma)
results_109 = model_109(G, Q, Q_L, epsilon_L, epsilon_L_F, V, t_g, T)
results_110 = model_110(G, Q, Q_L, epsilon_L, epsilon_L_F, Q_R, epsilon_RF, gamma)
results_111 = model_111(G, Q, Q_L, epsilon_L, epsilon_L_F, Q_R, epsilon_RF, V, t_g, T)


# Prepare Data for Plotting and Comparison
time_109, C_rise_109, C_decay_109 = results_109
time_111, C_rise_111, C_decay_111 = results_111

results_109_df = pd.DataFrame({
    'Time': np.concatenate([time_109, time_109]),
    'Concentration': np.concatenate([C_rise_109, C_decay_109]),
    'Model': 'Model 109'
}).dropna()

results_111_df = pd.DataFrame({
    'Time': np.concatenate([time_111, time_111]),
    'Concentration': np.concatenate([C_rise_111, C_decay_111]),
    'Model': 'Model 111'
}).dropna()


# Plotting
plt.figure(figsize=(10, 6))
plt.plot(results_109_df['Time'], results_109_df['Concentration'], label='Model 109')
plt.plot(results_111_df['Time'], results_111_df['Concentration'], label='Model 111', linestyle='dashed')
plt.title('Comparison of Model 109 and Model 111')
plt.xlabel('Time (minutes)')
plt.ylabel('Concentration (mg/m^3)')
plt.legend()
plt.grid(True)
plt.show()

# Calculate Mean Squared Error and R^2
mse = np.mean((results_109_df['Concentration'] - results_111_df['Concentration'])**2)
r2 = np.corrcoef(results_109_df['Concentration'], results_111_df['Concentration'])[0, 1]**2

# Normality Test
normality_test_109 = stats.shapiro(results_109_df['Concentration'])
normality_test_111 = stats.shapiro(results_111_df['Concentration'])

# Wilcoxon Signed-Rank Test
wilcoxon_test = stats.wilcoxon(results_109_df['Concentration'], results_111_df['Concentration'])

mse, r2, normality_test_109, normality_test_111, wilcoxon_test


# Prepare Data for Plotting and Comparison
# Convert all model results to DataFrame in a consistent format

# Combine all data for visualization
all_data_df = pd.concat([
    results_101_df,
    results_103_df,
    results_105_df,
    results_107_df,
    results_109_df,
    results_111_df
])


# Plotting all models
plt.figure(figsize=(12, 6))
for model in all_data_df['Model'].unique():
    subset = all_data_df[all_data_df['Model'] == model]
    plt.plot(subset['Time'], subset['Concentration'], label=model)
plt.title('Overall Model Comparison')
plt.xlabel('Time (minutes)')
plt.ylabel('Concentration (mg/m^3)')
plt.legend()
plt.grid(True)
plt.show()
