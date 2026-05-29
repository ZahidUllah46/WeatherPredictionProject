import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("   WEATHER PREDICTION USING BACKPROPAGATION")
print("   Neural Network From Scratch")
print("=" * 60)

# -----------------------------
# LOAD DATASET
# -----------------------------
data = pd.read_csv("Project 1 - Weather Dataset.csv")

print(f"\n✅ Dataset Loaded: {data.shape[0]} rows, {data.shape[1]} columns")

# Remove unnecessary column
if "Date/Time" in data.columns:
    data = data.drop(columns=["Date/Time"])

# Convert weather labels into numbers
encoder = LabelEncoder()
data["Weather"] = encoder.fit_transform(data["Weather"])

# ✅ FIX: weather_classes directly from encoder
weather_classes = encoder.classes_
num_classes = len(weather_classes)

print(f"\n✅ Weather Classes Found: {num_classes}")
print(f"   Classes: {list(weather_classes)}")

# Input and Output
X = data.drop(columns=["Weather"]).values
y = data["Weather"].values

# Normalize
scaler = MinMaxScaler()
X = scaler.fit_transform(X)

print(f"\n✅ Data Prepared:")
print(f"   Input Features : {X.shape[1]}")
print(f"   Total Samples  : {X.shape[0]}")
print(f"   Total Classes  : {num_classes}")

# One-Hot Encode
def one_hot_encode(y, num_classes):
    result = np.zeros((len(y), num_classes))
    for i, val in enumerate(y):
        result[i][int(val)] = 1
    return result

y_encoded = one_hot_encode(y, num_classes)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42
)

y_train_labels = np.argmax(y_train, axis=1)
y_test_labels  = np.argmax(y_test,  axis=1)

print(f"\n✅ Data Split:")
print(f"   Training Samples : {X_train.shape[0]}")
print(f"   Testing Samples  : {X_test.shape[0]}")

# -----------------------------
# NEURAL NETWORK SETUP
# -----------------------------

input_neurons  = X_train.shape[1]
hidden_neurons = 10
output_neurons = num_classes   # ✅ FIX: exact number of classes

np.random.seed(42)

# Xavier Initialization
weights_input_hidden  = np.random.randn(input_neurons, hidden_neurons) * np.sqrt(2.0 / input_neurons)
weights_hidden_output = np.random.randn(hidden_neurons, output_neurons) * np.sqrt(2.0 / hidden_neurons)

bias_hidden = np.zeros((1, hidden_neurons))
bias_output = np.zeros((1, output_neurons))

learning_rate = 0.1

print(f"\n✅ Neural Network Architecture:")
print(f"   Input Layer  : {input_neurons} neurons")
print(f"   Hidden Layer : {hidden_neurons} neurons")
print(f"   Output Layer : {output_neurons} neurons")
print(f"   Learning Rate: {learning_rate}")

# -----------------------------
# ACTIVATION FUNCTIONS
# -----------------------------

def sigmoid(x):
    x = np.clip(x, -500, 500)
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

# -----------------------------
# FORWARD PROPAGATION
# -----------------------------

def forward_propagation(X):
    hidden_input  = np.dot(X, weights_input_hidden) + bias_hidden
    hidden_output = sigmoid(hidden_input)
    final_input      = np.dot(hidden_output, weights_hidden_output) + bias_output
    predicted_output = softmax(final_input)
    return hidden_output, predicted_output

# -----------------------------
# TRAINING
# -----------------------------

epochs        = 2000
error_list    = []
accuracy_list = []

print("\n" + "=" * 60)
print("   TRAINING STARTED")
print("=" * 60)

for epoch in range(epochs):

    # FORWARD PROPAGATION
    hidden_output, predicted_output = forward_propagation(X_train)

    # ERROR CALCULATION
    epsilon = 1e-15
    loss = -np.mean(np.sum(y_train * np.log(predicted_output + epsilon), axis=1))
    error_list.append(loss)

    # Training Accuracy
    train_preds    = np.argmax(predicted_output, axis=1)
    train_accuracy = np.mean(train_preds == y_train_labels) * 100
    accuracy_list.append(train_accuracy)

    # BACKPROPAGATION
    output_error = predicted_output - y_train
    d_output     = output_error / len(X_train)

    hidden_error = d_output.dot(weights_hidden_output.T)
    d_hidden     = hidden_error * sigmoid_derivative(hidden_output)

    # WEIGHT UPDATES
    weights_hidden_output -= hidden_output.T.dot(d_output) * learning_rate
    weights_input_hidden  -= X_train.T.dot(d_hidden) * learning_rate
    bias_output           -= np.sum(d_output, axis=0, keepdims=True) * learning_rate
    bias_hidden           -= np.sum(d_hidden, axis=0, keepdims=True) * learning_rate

    if epoch % 200 == 0:
        print(f"   Epoch {epoch:4d} | Loss: {loss:.4f} | Accuracy: {train_accuracy:.2f}%")

print("\n✅ Training Complete!")

# -----------------------------
# TESTING
# -----------------------------

_, test_predictions = forward_propagation(X_test)
predicted_labels    = np.argmax(test_predictions, axis=1)
test_accuracy       = np.mean(predicted_labels == y_test_labels) * 100

print("\n" + "=" * 60)
print("   FINAL RESULTS")
print("=" * 60)
print(f"\n   ✅ Test Accuracy : {test_accuracy:.2f}%")
print(f"   ✅ Final Loss    : {error_list[-1]:.4f}")
print(f"   ✅ Total Epochs  : {epochs}")

# ✅ FIX: Classification Report — labels list se match karwaya
unique_labels = np.unique(np.concatenate([y_test_labels, predicted_labels]))
print("\n" + "=" * 60)
print("   CLASSIFICATION REPORT")
print("=" * 60)
print(classification_report(
    y_test_labels,
    predicted_labels,
    labels=unique_labels,
    target_names=weather_classes[unique_labels]
))

# -----------------------------
# CONFUSION MATRIX
# -----------------------------

cm = confusion_matrix(y_test_labels, predicted_labels, labels=unique_labels)
short_labels = [w[:8] for w in weather_classes[unique_labels]]

# -----------------------------
# VISUALIZATION
# -----------------------------

fig = plt.figure(figsize=(18, 14))
fig.patch.set_facecolor('#0d1117')
gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

GRID_COLOR = '#21262d'
TEXT_COLOR = '#e6edf3'
ACCENT1    = '#58a6ff'
ACCENT2    = '#3fb950'
ACCENT3    = '#ff7b72'

plt.rcParams.update({
    'text.color':      TEXT_COLOR,
    'axes.labelcolor': TEXT_COLOR,
    'xtick.color':     TEXT_COLOR,
    'ytick.color':     TEXT_COLOR,
})

# 1. Training Loss Curve
ax1 = fig.add_subplot(gs[0, 0])
ax1.set_facecolor('#161b22')
ax1.plot(error_list, color=ACCENT3, linewidth=2, label='Training Loss')
ax1.fill_between(range(len(error_list)), error_list, alpha=0.15, color=ACCENT3)
ax1.set_title('Training Loss Curve', color=TEXT_COLOR, fontsize=13, fontweight='bold', pad=10)
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Loss (Cross-Entropy)')
ax1.legend(facecolor='#21262d', labelcolor=TEXT_COLOR)
ax1.grid(True, color=GRID_COLOR, linewidth=0.6)
for spine in ax1.spines.values():
    spine.set_edgecolor(GRID_COLOR)

# 2. Training Accuracy Curve
ax2 = fig.add_subplot(gs[0, 1])
ax2.set_facecolor('#161b22')
ax2.plot(accuracy_list, color=ACCENT2, linewidth=2, label='Training Accuracy')
ax2.fill_between(range(len(accuracy_list)), accuracy_list, alpha=0.15, color=ACCENT2)
ax2.set_title('Training Accuracy Curve', color=TEXT_COLOR, fontsize=13, fontweight='bold', pad=10)
ax2.set_xlabel('Epochs')
ax2.set_ylabel('Accuracy (%)')
ax2.set_ylim([0, 105])
ax2.legend(facecolor='#21262d', labelcolor=TEXT_COLOR)
ax2.grid(True, color=GRID_COLOR, linewidth=0.6)
for spine in ax2.spines.values():
    spine.set_edgecolor(GRID_COLOR)

# 3. Confusion Matrix
ax3 = fig.add_subplot(gs[0, 2])
ax3.set_facecolor('#161b22')
im = ax3.imshow(cm, interpolation='nearest', cmap='Blues')
ax3.set_title('Confusion Matrix', color=TEXT_COLOR, fontsize=13, fontweight='bold', pad=10)
tick_marks = np.arange(len(short_labels))
ax3.set_xticks(tick_marks)
ax3.set_yticks(tick_marks)
ax3.set_xticklabels(short_labels, rotation=45, ha='right', fontsize=7)
ax3.set_yticklabels(short_labels, fontsize=7)
ax3.set_xlabel('Predicted Label')
ax3.set_ylabel('True Label')
thresh = cm.max() / 2.0
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax3.text(j, i, str(cm[i, j]), ha='center', va='center',
                 color='white' if cm[i, j] > thresh else 'black', fontsize=7)
plt.colorbar(im, ax=ax3)
for spine in ax3.spines.values():
    spine.set_edgecolor(GRID_COLOR)

# 4. Per-Class Accuracy
ax4 = fig.add_subplot(gs[1, 0])
ax4.set_facecolor('#161b22')
per_class_acc = []
for i in unique_labels:
    mask = y_test_labels == i
    acc  = np.mean(predicted_labels[mask] == i) * 100 if mask.sum() > 0 else 0
    per_class_acc.append(acc)

colors_bar = plt.cm.Set2(np.linspace(0, 1, len(short_labels)))
bars = ax4.bar(short_labels, per_class_acc, color=colors_bar, edgecolor=GRID_COLOR, linewidth=0.8)
ax4.set_title('Per-Class Accuracy', color=TEXT_COLOR, fontsize=13, fontweight='bold', pad=10)
ax4.set_xlabel('Weather Class')
ax4.set_ylabel('Accuracy (%)')
ax4.set_ylim([0, 115])
ax4.tick_params(axis='x', rotation=45)
for bar, acc in zip(bars, per_class_acc):
    ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
             f'{acc:.1f}%', ha='center', va='bottom', fontsize=8, color=TEXT_COLOR)
ax4.grid(True, axis='y', color=GRID_COLOR, linewidth=0.6)
for spine in ax4.spines.values():
    spine.set_edgecolor(GRID_COLOR)

# 5. Actual vs Predicted
ax5 = fig.add_subplot(gs[1, 1])
ax5.set_facecolor('#161b22')
n_show = min(50, len(y_test_labels))
x_idx  = np.arange(n_show)
ax5.plot(x_idx, y_test_labels[:n_show],    'o', color=ACCENT1, markersize=4, label='Actual',    alpha=0.8)
ax5.plot(x_idx, predicted_labels[:n_show], 'x', color=ACCENT3, markersize=5, label='Predicted', alpha=0.8)
ax5.set_title('Actual vs Predicted (50 Samples)', color=TEXT_COLOR, fontsize=13, fontweight='bold', pad=10)
ax5.set_xlabel('Sample Index')
ax5.set_ylabel('Class Label')
ax5.legend(facecolor='#21262d', labelcolor=TEXT_COLOR)
ax5.grid(True, color=GRID_COLOR, linewidth=0.6)
for spine in ax5.spines.values():
    spine.set_edgecolor(GRID_COLOR)

# 6. Summary Card
ax6 = fig.add_subplot(gs[1, 2])
ax6.set_facecolor('#161b22')
ax6.axis('off')
summary_text = (
    f"  SUMMARY RESULTS\n"
    f"{'─'*30}\n\n"
    f"  Model  : Backpropagation NN\n"
    f"  Topic  : Weather Prediction\n\n"
    f"  Architecture\n"
    f"  Input  : {input_neurons} neurons\n"
    f"  Hidden : {hidden_neurons} neurons\n"
    f"  Output : {output_neurons} neurons\n\n"
    f"  Training\n"
    f"  Epochs : {epochs}\n"
    f"  LR     : {learning_rate}\n\n"
    f"  Results\n"
    f"  Test Accuracy : {test_accuracy:.2f}%\n"
    f"  Final Loss    : {error_list[-1]:.4f}\n"
    f"  Classes       : {num_classes}\n"
)
ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes,
         fontsize=10, verticalalignment='top', fontfamily='monospace',
         color=ACCENT2,
         bbox=dict(boxstyle='round,pad=0.8', facecolor='#0d1117', edgecolor=ACCENT1, linewidth=1.5))

fig.suptitle(
    'Weather Prediction — Backpropagation Neural Network',
    fontsize=16, fontweight='bold', color=TEXT_COLOR, y=0.98
)

plt.savefig('backpropagation_results.png', dpi=150, bbox_inches='tight',
            facecolor=fig.get_facecolor())
plt.show()

print("\n✅ Graph saved as: backpropagation_results.png")
print("\n" + "=" * 60)
print("   Zahid your project has been completed congratulations")
print("=" * 6000)
