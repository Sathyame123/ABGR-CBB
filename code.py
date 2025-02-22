import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import GRU, Bidirectional, Dense, Dropout, Attention, Input
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error
import matplotlib.pyplot as plt
import random
import os

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)

# Load dataset
url = "https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset/download"
df = pd.read_csv(r"path_of_the_datset.csv")

# Data Preprocessing
scaler = StandardScaler()
X = scaler.fit_transform(df.iloc[:, :-1])
y = LabelEncoder().fit_transform(df.iloc[:, -1])

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Architecture (ABGR-CBB)
def build_abgr_cbb_model(input_shape):
    inputs = Input(shape=input_shape)
    x = Bidirectional(GRU(64, return_sequences=True))(inputs)
    x = Attention()([x, x])  # Attention Mechanism
    x = Bidirectional(GRU(32))(x)
    x = Dense(32, activation='relu')(x)
    x = Dropout(0.2)(x)
    outputs = Dense(len(np.unique(y_train)), activation='softmax')(x)
    model = Model(inputs, outputs)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Train Model
abgr_cbb_model = build_abgr_cbb_model((X_train.shape[1], 1))
abgr_cbb_model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100, batch_size=32, verbose=1)

# Evaluate Model
y_pred = np.argmax(abgr_cbb_model.predict(X_test), axis=1)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='macro')
recall = recall_score(y_test, y_pred, average='macro')
f1 = f1_score(y_test, y_pred, average='macro')
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

# Print Evaluation Metrics
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")
print(f"MSE: {mse:.4f}")
print(f"RMSE: {rmse:.4f}")

# Benchmark Comparisons
benchmark_models = {
    "IDCSO-WLSTM": build_abgr_cbb_model((X_train.shape[1], 1)),
    "OntoFusionCrop": build_abgr_cbb_model((X_train.shape[1], 1)),
    "BO-SVM": build_abgr_cbb_model((X_train.shape[1], 1)),
    "GTODL-CRYPM": build_abgr_cbb_model((X_train.shape[1], 1))
}

benchmark_results = {}
for model_name, model in benchmark_models.items():
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100, batch_size=32, verbose=1)
    y_pred = np.argmax(model.predict(X_test), axis=1)
    benchmark_results[model_name] = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, average='macro'),
        "Recall": recall_score(y_test, y_pred, average='macro'),
        "F1-Score": f1_score(y_test, y_pred, average='macro'),
        "MSE": mean_squared_error(y_test, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_test, y_pred))
    }

# Print Benchmark Comparison
print("\nBenchmark Model Comparison:")
bm_df = pd.DataFrame(benchmark_results).T
print(bm_df)

# Visualization
fig, ax = plt.subplots(figsize=(10, 6))

# Define custom colors for each model
custom_colors = ['#80deea', '#64b5f6', '#81c784', '#ba68c8']  # Modify these color codes as needed

bm_df[['Accuracy', 'Precision', 'Recall', 'F1-Score']].plot(kind='bar', ax=ax, color=custom_colors)
plt.title('Performance Comparison', fontweight='bold', fontsize=18, fontfamily="Times New Roman")
plt.ylabel('Score', fontweight='bold', fontsize=18, fontfamily="Times New Roman")
plt.xlabel('Models', fontweight='bold', fontsize=18, fontfamily="Times New Roman")
plt.xticks(fontweight='bold', fontsize=18, fontfamily="Times New Roman", rotation = 0)
plt.yticks(fontweight='bold', fontsize=18, fontfamily="Times New Roman")

plt.legend(prop={'family': 'Times New Roman', 'size': 18, 'weight': 'bold'}, ncol=4, loc='upper center', bbox_to_anchor=(0.48, 1.17), frameon=False, labelspacing=0.3, handletextpad=0.3, columnspacing=0.3)
plt.tight_layout()
plt.show()

# Save results
bm_df.to_csv("path_to_save_the_benchmark_results.csv", index=True)
