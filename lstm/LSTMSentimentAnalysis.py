import os
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM, Dropout

# ---------------------------------------------------------
# Configuration & Hyperparameters
# ---------------------------------------------------------
MAX_FEATURES = 10000    # Number of words to consider as features
MAX_LEN = 500           # Cut texts after this number of words (among top max_features)
BATCH_SIZE = 32
EMBEDDING_DIM = 32
RNN_UNITS = 64
EPOCHS = 3

def run_experiment():
    print("----------------------------------------------------------------")
    print("Starting LSTM Experiment for Sentiment Analysis")
    print("----------------------------------------------------------------")

    # 1. Load and Preprocess Data
    print('\n[1] Loading IMDB data...')
    # input_train/test are lists of integers (word indices)
    # y_train/test are binary labels (0 = negative, 1 = positive)
    (input_train, y_train), (input_test, y_test) = imdb.load_data(num_words=MAX_FEATURES)
    
    print(f'    Train sequences: {len(input_train)}')
    print(f'    Test sequences:  {len(input_test)}')

    print('\n[2] Padding sequences (samples x time)...')
    # Pad sequences to ensure uniform length for the model input
    input_train = sequence.pad_sequences(input_train, maxlen=MAX_LEN)
    input_test = sequence.pad_sequences(input_test, maxlen=MAX_LEN)
    
    print(f'    input_train shape: {input_train.shape}')
    print(f'    input_test shape:  {input_test.shape}')

    # 2. Build the LSTM Model
    # We replace SimpleRNN with LSTM cells to handle long-term dependencies better
    print('\n[3] Building LSTM Model...')
    model = Sequential()

    # Embedding layer: Turns positive integers (indexes) into dense vectors of fixed size
    model.add(Embedding(input_dim=MAX_FEATURES, output_dim=EMBEDDING_DIM, input_length=MAX_LEN))

    # LSTM Layer 1: 
    # return_sequences=True is crucial here to pass the sequence to the next recurrent layer
    model.add(LSTM(RNN_UNITS, return_sequences=True))
    model.add(Dropout(0.2)) # Dropout for regularization

    # LSTM Layer 2: 
    # return_sequences=False (default) because we only need the final state for classification
    model.add(LSTM(RNN_UNITS))
    model.add(Dropout(0.2))

    # Output layer: Sigmoid for binary classification
    model.add(Dense(1, activation='sigmoid'))

    model.summary()

    # 3. Compile
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    # 4. Train (Fit)
    print('\n[4] Training Model...')
    # Using validation_split to monitor validation accuracy during training
    history = model.fit(input_train, y_train,
                        epochs=EPOCHS,
                        batch_size=BATCH_SIZE,
                        validation_split=0.2,
                        verbose=1)

    # 5. Evaluate
    print('\n[5] Evaluating on Test Set...')
    loss, acc = model.evaluate(input_test, y_test, batch_size=BATCH_SIZE)
    
    print("\n----------------------------------------------------------------")
    print(f"Final Test Accuracy: {acc:.4f}")
    print(f"Final Test Loss:     {loss:.4f}")
    print("----------------------------------------------------------------")

if __name__ == "__main__":
    # Ensure directories exist if this script generates logs/outputs (optional)
    run_experiment()
