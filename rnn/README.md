# ML Fundamentals – Recurrent Neural Networks (RNN)

# Session Summary: RNN Implementation for IMDB Sentiment Analysis

## What I Did During the Session

1.  **Implemented a Basic RNN for Sentiment Analysis**
    - Created a Python script (`RNNImdbSentimentAnalysis.py`) using TensorFlow and Keras
    - Utilized the `SimpleRNN` layer to process sequential data
    - Configured binary classification architecture with a sigmoid output layer

2.  **Transitioned to Deep RNN Architecture**
    - Upgraded the model to use Long Short-Term Memory (`LSTM`) layers
    - Implemented a Stacked LSTM architecture with multiple layers
    - Configured the first LSTM layer with `return_sequences=True` to pass full sequences to the next layer
    - Added `Dropout` layers to prevent overfitting in the deep network

3.  **Explored the IMDB Dataset Structure**
    - Loaded and inspected the standardized IMDB movie review dataset
    - Applied sequence padding (`pad_sequences`) to ensure uniform input lengths
    - Utilized integer encoding where words are mapped to frequency-based indices

4.  **Analyzed Data Flow in Recurrent Networks**
    - Examined how data propagates through time steps ($t_0$ to $t_N$) and through layers (depth)
    - Clarified the difference between `return_sequences=True` (passing full history) vs `False` (passing only the final state)
    - Understood how the final Dense layer classifies the "thought vector" from the last time step

## What I Learned

1.  **RNN vs. LSTM Architectures**:
    - `SimpleRNN` is rarely used in deep networks due to vanishing gradients
    - `LSTM` units are superior for capturing long-term dependencies in text
    - Stacking layers allows the network to learn more abstract temporal features

2.  **Sequence Processing Mechanics**:
    - How `return_sequences=True` is required when stacking RNN layers to maintain the temporal dimension
    - How `return_sequences=False` acts as a summary operation, condensing a sequence into a single vector
    - The concept of "Unrolling through time": how the network processes inputs step-by-step while maintaining an internal state

3.  **NLP Data Preparation**:
    - The importance of Integer Encoding for representing words in neural networks
    - The role of the `Embedding` layer in converting integer indices into dense, meaningful vectors
    - How padding works to handle variable-length sequences in batch processing

4.  **Many-to-One Architecture**:
    - Why sentiment analysis is typically a "Many-to-One" task (sequence in -> single decision out)
    - How the final classification layer (`Dense`) only needs to see the final output of the RNN to make a decision for the whole text

This session provided a deep dive into the mechanics of Recurrent Neural Networks, moving from simple implementations to robust, deep LSTM architectures for practical natural language processing tasks.
