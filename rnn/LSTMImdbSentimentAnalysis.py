import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM, Dropout

# ---------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------
max_features = 10000  # Number of words to consider as features
maxlen = 500          # Cut texts after this number of words
batch_size = 32
embedding_dim = 32
rnn_units = 64
epochs = 3

# ---------------------------------------------------------
# 1. Load and Preprocess Data
# ---------------------------------------------------------
print('Loading data...')
# input_train and input_test are lists of integers (word indices)
(input_train, y_train), (input_test, y_test) = imdb.load_data(num_words=max_features)
print(f'{len(input_train)} train sequences')
print(f'{len(input_test)} test sequences')

print('Pad sequences (samples x time)')
# Pad sequences to ensure uniform length for the model
input_train = sequence.pad_sequences(input_train, maxlen=maxlen)
input_test = sequence.pad_sequences(input_test, maxlen=maxlen)
print(f'input_train shape: {input_train.shape}')
print(f'input_test shape: {input_test.shape}')

# ---------------------------------------------------------
# 2. Build the Model
# ---------------------------------------------------------
print('Build model...')
model = Sequential()

# Embedding layer turns positive integers (indexes) into dense vectors of fixed size
model.add(Embedding(max_features, embedding_dim))

# 1st LSTM layer: Returns sequences so the next LSTM layer has data for every time step
model.add(LSTM(rnn_units, return_sequences=True))
model.add(Dropout(0.2))

# 2nd LSTM layer: Default return_sequences=False, so it only returns the final vector (Deep "Many-to-One")
model.add(LSTM(rnn_units))
model.add(Dropout(0.2))

# Output layer: Sigmoid activation for binary classification (0 or 1)
model.add(Dense(1, activation='sigmoid'))

model.summary()

# ---------------------------------------------------------
# 3. Compile and Train
# ---------------------------------------------------------
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['acc'])

print('Train...')
history = model.fit(input_train, y_train,
                    epochs=epochs,
                    batch_size=128,
                    validation_split=0.2)

# ---------------------------------------------------------
# 4. Evaluate
# ---------------------------------------------------------
print('Evaluating on test set...')
loss, acc = model.evaluate(input_test, y_test, batch_size=batch_size)
print(f'Test accuracy: {acc:.4f}')
