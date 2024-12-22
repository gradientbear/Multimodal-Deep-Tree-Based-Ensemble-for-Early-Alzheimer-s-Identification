from matplotlib import pyplot as plt
import seaborn as sns
from preprocess import get_data, all_features, data_models
from imblearn.over_sampling import SMOTE

# Load data
data = get_data()

def apply_smote(data, features, target):
    sm = SMOTE(sampling_strategy="not majority")
    resampled_data, resampled_target = sm.fit_resample(data[features], data[target])
    resampled_data[target] = resampled_target
    return resampled_data

def plot_data(data, x_feature, y_feature, target, title):
    fig, ax = plt.subplots(figsize=(12, 8))
    scatter = ax.scatter(data[x_feature], data[y_feature], c=data[target], s=40, edgecolors='none', cmap='viridis')
    legend_plt = ax.legend(*scatter.legend_elements(), loc="lower left", title="Digits")
    ax.add_artist(legend_plt)
    plt.title(title)
    plt.show()

# Apply SMOTE
print("Original dataset shape {}".format(data.shape))
resampled_data = apply_smote(data, all_features, 'diagnosis')
print("Resampled dataset shape {}".format(resampled_data.shape))

# Plotting the data before and after SMOTE
plot_data(data, 'ravlt_immediate', 'adas_memory', 'diagnosis', 'Before SMOTE')
plot_data(resampled_data, 'ravlt_immediate', 'adas_memory', 'diagnosis', 'After SMOTE')

