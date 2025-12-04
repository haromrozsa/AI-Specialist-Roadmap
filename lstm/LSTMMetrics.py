"""
RNN and LSTM Experiments for IMDB Sentiment Analysis
=====================================================
This script trains multiple RNN architectures and evaluates them with comprehensive metrics.
"""

import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, SimpleRNN, LSTM, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import time

# ---------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------
MAX_FEATURES = 10000  # Number of words to consider as features
MAX_LEN = 200  # Cut texts after this number of words (reduced for faster training)
BATCH_SIZE = 64
EMBEDDING_DIM = 128
EPOCHS = 10
VALIDATION_SPLIT = 0.2


# ---------------------------------------------------------
# Data Loading and Preprocessing
# ---------------------------------------------------------
def load_and_preprocess_data():
    """Load and preprocess IMDB dataset"""
    print('=' * 70)
    print('LOADING AND PREPROCESSING DATA')
    print('=' * 70)

    print('Loading IMDB data...')
    (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=MAX_FEATURES)

    print(f'Train sequences: {len(x_train)}')
    print(f'Test sequences:  {len(x_test)}')

    print('\nPadding sequences...')
    x_train = sequence.pad_sequences(x_train, maxlen=MAX_LEN)
    x_test = sequence.pad_sequences(x_test, maxlen=MAX_LEN)

    print(f'x_train shape: {x_train.shape}')
    print(f'x_test shape:  {x_test.shape}')

    return (x_train, y_train), (x_test, y_test)


# ---------------------------------------------------------
# Model Architectures
# ---------------------------------------------------------
def create_simple_rnn_model():
    """Create a simple RNN model"""
    model = Sequential([
        Embedding(MAX_FEATURES, EMBEDDING_DIM, input_length=MAX_LEN),
        SimpleRNN(64),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    return model


def create_single_lstm_model():
    """Create a single LSTM layer model"""
    model = Sequential([
        Embedding(MAX_FEATURES, EMBEDDING_DIM, input_length=MAX_LEN),
        LSTM(64),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    return model


def create_stacked_lstm_model():
    """Create a stacked LSTM model (2 layers)"""
    model = Sequential([
        Embedding(MAX_FEATURES, EMBEDDING_DIM, input_length=MAX_LEN),
        LSTM(64, return_sequences=True),
        Dropout(0.3),
        LSTM(32),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])
    return model


def create_bidirectional_lstm_model():
    """Create a bidirectional LSTM model"""
    model = Sequential([
        Embedding(MAX_FEATURES, EMBEDDING_DIM, input_length=MAX_LEN),
        Bidirectional(LSTM(64)),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    return model


def create_deep_lstm_model():
    """Create a deep LSTM model (3 layers)"""
    model = Sequential([
        Embedding(MAX_FEATURES, EMBEDDING_DIM, input_length=MAX_LEN),
        LSTM(128, return_sequences=True),
        Dropout(0.3),
        LSTM(64, return_sequences=True),
        Dropout(0.3),
        LSTM(32),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])
    return model


# ---------------------------------------------------------
# Training and Evaluation
# ---------------------------------------------------------
def compile_and_train_model(model, model_name, x_train, y_train, x_test, y_test):
    """Compile and train a model with callbacks"""
    print('\n' + '=' * 70)
    print(f'TRAINING: {model_name}')
    print('=' * 70)

    # Compile with comprehensive metrics
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=[
            'accuracy',
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall'),
            tf.keras.metrics.AUC(name='auc')
        ]
    )

    # Display model architecture
    model.summary()

    # Callbacks
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights=True,
        verbose=1
    )

    # Train the model
    start_time = time.time()
    history = model.fit(
        x_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=VALIDATION_SPLIT,
        callbacks=[early_stopping],
        verbose=1
    )
    training_time = time.time() - start_time

    # Evaluate on test set
    print('\n' + '-' * 70)
    print('EVALUATING ON TEST SET')
    print('-' * 70)

    test_results = model.evaluate(x_test, y_test, batch_size=BATCH_SIZE, verbose=0)

    # Extract metrics
    test_loss = test_results[0]
    test_accuracy = test_results[1]
    test_precision = test_results[2]
    test_recall = test_results[3]
    test_auc = test_results[4]

    # Calculate F1 score
    test_f1 = 2 * (test_precision * test_recall) / (test_precision + test_recall + 1e-7)

    print(f'\nTest Loss:      {test_loss:.4f}')
    print(f'Test Accuracy:  {test_accuracy:.4f}')
    print(f'Test Precision: {test_precision:.4f}')
    print(f'Test Recall:    {test_recall:.4f}')
    print(f'Test AUC:       {test_auc:.4f}')
    print(f'Test F1-Score:  {test_f1:.4f}')
    print(f'Training Time:  {training_time:.2f} seconds')

    # Get validation metrics from history
    final_val_accuracy = history.history['val_accuracy'][-1]
    final_val_loss = history.history['val_loss'][-1]

    print(f'\nFinal Validation Accuracy: {final_val_accuracy:.4f}')
    print(f'Final Validation Loss:     {final_val_loss:.4f}')

    return model, history, {
        'model_name': model_name,
        'test_loss': test_loss,
        'test_accuracy': test_accuracy,
        'test_precision': test_precision,
        'test_recall': test_recall,
        'test_auc': test_auc,
        'test_f1': test_f1,
        'val_accuracy': final_val_accuracy,
        'val_loss': final_val_loss,
        'training_time': training_time
    }


# ---------------------------------------------------------
# Visualization Functions
# ---------------------------------------------------------
def plot_training_history(history, model_name):
    """Plot training and validation metrics"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'Training History: {model_name}', fontsize=16, fontweight='bold')

    metrics = ['accuracy', 'loss', 'precision', 'recall']
    titles = ['Accuracy', 'Loss', 'Precision', 'Recall']

    for idx, (metric, title) in enumerate(zip(metrics, titles)):
        row = idx // 2
        col = idx % 2
        ax = axes[row, col]

        if metric in history.history:
            epochs = range(1, len(history.history[metric]) + 1)
            ax.plot(epochs, history.history[metric], 'b-', label=f'Training {title}')
            ax.plot(epochs, history.history[f'val_{metric}'], 'r-', label=f'Validation {title}')
            ax.set_xlabel('Epoch')
            ax.set_ylabel(title)
            ax.set_title(f'{title} over Epochs')
            ax.legend()
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{model_name.replace(" ", "_")}_training_history.png', dpi=300, bbox_inches='tight')
    plt.show()


def plot_confusion_matrix(y_true, y_pred, model_name):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Negative', 'Positive'],
                yticklabels=['Negative', 'Positive'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(f'Confusion Matrix: {model_name}')
    plt.savefig(f'{model_name.replace(" ", "_")}_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()


def compare_models(results_list):
    """Create comparison visualizations for all models"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Model Comparison', fontsize=18, fontweight='bold')

    model_names = [r['model_name'] for r in results_list]

    # Accuracy comparison
    ax1 = axes[0, 0]
    test_acc = [r['test_accuracy'] for r in results_list]
    val_acc = [r['val_accuracy'] for r in results_list]
    x = np.arange(len(model_names))
    width = 0.35
    ax1.bar(x - width / 2, test_acc, width, label='Test Accuracy', color='steelblue')
    ax1.bar(x + width / 2, val_acc, width, label='Validation Accuracy', color='coral')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Accuracy Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels(model_names, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')

    # Loss comparison
    ax2 = axes[0, 1]
    test_loss = [r['test_loss'] for r in results_list]
    val_loss = [r['val_loss'] for r in results_list]
    ax2.bar(x - width / 2, test_loss, width, label='Test Loss', color='steelblue')
    ax2.bar(x + width / 2, val_loss, width, label='Validation Loss', color='coral')
    ax2.set_ylabel('Loss')
    ax2.set_title('Loss Comparison')
    ax2.set_xticks(x)
    ax2.set_xticklabels(model_names, rotation=45, ha='right')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')

    # Precision, Recall, F1 comparison
    ax3 = axes[1, 0]
    precision = [r['test_precision'] for r in results_list]
    recall = [r['test_recall'] for r in results_list]
    f1 = [r['test_f1'] for r in results_list]
    x_pos = np.arange(len(model_names))
    width = 0.25
    ax3.bar(x_pos - width, precision, width, label='Precision', color='lightgreen')
    ax3.bar(x_pos, recall, width, label='Recall', color='lightcoral')
    ax3.bar(x_pos + width, f1, width, label='F1-Score', color='lightskyblue')
    ax3.set_ylabel('Score')
    ax3.set_title('Precision, Recall, F1-Score Comparison')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(model_names, rotation=45, ha='right')
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')

    # Training time comparison
    ax4 = axes[1, 1]
    training_times = [r['training_time'] for r in results_list]
    ax4.bar(model_names, training_times, color='mediumpurple')
    ax4.set_ylabel('Time (seconds)')
    ax4.set_title('Training Time Comparison')
    ax4.set_xticklabels(model_names, rotation=45, ha='right')
    ax4.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()


def print_classification_report(model, x_test, y_test, model_name):
    """Print detailed classification report"""
    print('\n' + '=' * 70)
    print(f'DETAILED CLASSIFICATION REPORT: {model_name}')
    print('=' * 70)

    y_pred_probs = model.predict(x_test, batch_size=BATCH_SIZE, verbose=0)
    y_pred = (y_pred_probs > 0.5).astype('int32')

    print(classification_report(y_test, y_pred,
                                target_names=['Negative (0)', 'Positive (1)'],
                                digits=4))

    return y_pred


# ---------------------------------------------------------
# Main Execution
# ---------------------------------------------------------
def main():
    """Main execution function"""
    print('\n')
    print('*' * 70)
    print('RNN AND LSTM EXPERIMENTS FOR SENTIMENT ANALYSIS')
    print('*' * 70)
    print('\n')

    # Load data
    (x_train, y_train), (x_test, y_test) = load_and_preprocess_data()

    # Define experiments
    experiments = [
        ('Simple RNN', create_simple_rnn_model),
        ('Single LSTM', create_single_lstm_model),
        ('Stacked LSTM', create_stacked_lstm_model),
        ('Bidirectional LSTM', create_bidirectional_lstm_model),
        ('Deep LSTM', create_deep_lstm_model),
    ]

    results_list = []

    # Run each experiment
    for model_name, model_fn in experiments:
        model = model_fn()
        model, history, results = compile_and_train_model(
            model, model_name, x_train, y_train, x_test, y_test
        )

        # Visualizations
        plot_training_history(history, model_name)

        # Classification report and confusion matrix
        y_pred = print_classification_report(model, x_test, y_test, model_name)
        plot_confusion_matrix(y_test, y_pred, model_name)

        results_list.append(results)

        print('\n' + '=' * 70 + '\n')

    # Compare all models
    print('\n')
    print('*' * 70)
    print('FINAL COMPARISON OF ALL MODELS')
    print('*' * 70)
    print('\n')

    # Print summary table
    print(f"{'Model':<25} {'Test Acc':<10} {'Val Acc':<10} {'Test F1':<10} {'Time (s)':<10}")
    print('-' * 70)
    for r in results_list:
        print(
            f"{r['model_name']:<25} {r['test_accuracy']:<10.4f} {r['val_accuracy']:<10.4f} {r['test_f1']:<10.4f} {r['training_time']:<10.2f}")

    # Create comparison visualizations
    compare_models(results_list)

    # Find best model
    best_model_idx = np.argmax([r['test_accuracy'] for r in results_list])
    best_model = results_list[best_model_idx]

    print('\n' + '=' * 70)
    print('BEST MODEL')
    print('=' * 70)
    print(f"Model: {best_model['model_name']}")
    print(f"Test Accuracy: {best_model['test_accuracy']:.4f}")
    print(f"Validation Accuracy: {best_model['val_accuracy']:.4f}")
    print(f"Test F1-Score: {best_model['test_f1']:.4f}")
    print('=' * 70)

    print('\n✅ Experiments completed successfully!')


if __name__ == '__main__':
    main()