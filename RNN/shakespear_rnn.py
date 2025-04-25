import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import requests

def load_shakespeare_data():
    # Download Shakespeare text
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    text = requests.get(url).text
    return text

def preprocess_text(text, seq_length=40, is_char_level=True, max_samples=50000):
    if is_char_level:
        # Character-level tokenization
        chars = sorted(list(set(text)))
        char_to_idx = {char: idx for idx, char in enumerate(chars)}
        idx_to_char = {idx: char for idx, char in enumerate(chars)}
        
        # Create sequences
        sequences = []
        next_chars = []
        for i in range(0, min(len(text) - seq_length, max_samples)):
            sequences.append(text[i:i + seq_length])
            next_chars.append(text[i + seq_length])
            
        # Convert to numerical form
        X = np.zeros((len(sequences), seq_length, len(chars)), dtype=np.bool_)
        y = np.zeros((len(sequences), len(chars)), dtype=np.bool_)
        
        for i, sequence in enumerate(sequences):
            for t, char in enumerate(sequence):
                X[i, t, char_to_idx[char]] = 1
            y[i, char_to_idx[next_chars[i]]] = 1
            
        return X, y, char_to_idx, idx_to_char
    else:
        # Word-level tokenization
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts([text])
        sequences = tokenizer.texts_to_sequences([text])[0]
        
        # Create sequences
        X = []
        y = []
        for i in range(0, min(len(sequences) - seq_length, max_samples)):
            X.append(sequences[i:i + seq_length])
            y.append(sequences[i + seq_length])
            
        X = np.array(X)
        y = np.array(y)
        return X, y, tokenizer

def build_vanilla_rnn(vocab_size, seq_length):
    model = Sequential([
        SimpleRNN(64, input_shape=(seq_length, vocab_size), return_sequences=False),
        Dense(vocab_size, activation='softmax')
    ])
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    return model

def build_stacked_rnn(vocab_size, seq_length):
    model = Sequential([
        SimpleRNN(64, input_shape=(seq_length, vocab_size), return_sequences=True),
        SimpleRNN(64, return_sequences=False),
        Dense(vocab_size, activation='softmax')
    ])
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    return model

def build_bidirectional_rnn(vocab_size, seq_length):
    model = Sequential([
        Bidirectional(SimpleRNN(64, return_sequences=True), 
                     input_shape=(seq_length, vocab_size)),
        Bidirectional(SimpleRNN(64)),
        Dense(vocab_size, activation='softmax')
    ])
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    return model

def generate_text(model, seed_text, char_to_idx, idx_to_char, seq_length, num_chars=200):
    generated = seed_text
    for i in range(num_chars):
        x = np.zeros((1, seq_length, len(char_to_idx)))
        for t, char in enumerate(seed_text):
            x[0, t, char_to_idx[char]] = 1
            
        pred = model.predict(x, verbose=0)[0]
        next_idx = np.random.choice(len(pred), p=pred)
        next_char = idx_to_char[next_idx]
        
        generated += next_char
        seed_text = seed_text[1:] + next_char
        
    return generated

def main():
    # Load and preprocess data
    print("Loading Shakespeare text...")
    text = load_shakespeare_data()
    
    print("Preprocessing text...")
    seq_length = 40
    max_samples = 50000  # Limit the number of samples to speed up training
    X, y, char_to_idx, idx_to_char = preprocess_text(text, seq_length, max_samples=max_samples)
    vocab_size = len(char_to_idx)
    
    print(f"Training on {len(X)} samples with vocabulary size {vocab_size}")
    
    # Train Vanilla RNN
    print("\nTraining Vanilla RNN...")
    vanilla_rnn = build_vanilla_rnn(vocab_size, seq_length)
    vanilla_history = vanilla_rnn.fit(X, y, batch_size=128, epochs=5, validation_split=0.1)
    
    # Train Stacked RNN
    print("\nTraining Stacked RNN...")
    stacked_rnn = build_stacked_rnn(vocab_size, seq_length)
    stacked_history = stacked_rnn.fit(X, y, batch_size=128, epochs=5, validation_split=0.1)
    
    # Train Bidirectional RNN
    print("\nTraining Bidirectional RNN...")
    bidirectional_rnn = build_bidirectional_rnn(vocab_size, seq_length)
    bidirectional_history = bidirectional_rnn.fit(X, y, batch_size=128, epochs=5, validation_split=0.1)
    
    # Generate and compare text
    seed_text = text[:seq_length]
    
    print("\nGenerating text using Vanilla RNN:")
    vanilla_text = generate_text(vanilla_rnn, seed_text, char_to_idx, idx_to_char, seq_length, num_chars=200)
    print(vanilla_text)
    
    print("\nGenerating text using Stacked RNN:")
    stacked_text = generate_text(stacked_rnn, seed_text, char_to_idx, idx_to_char, seq_length, num_chars=200)
    print(stacked_text)
    
    print("\nGenerating text using Bidirectional RNN:")
    bidirectional_text = generate_text(bidirectional_rnn, seed_text, char_to_idx, idx_to_char, seq_length, num_chars=200)
    print(bidirectional_text)
    
    print("\nModel Comparison:")
    print("1. Vanilla RNN: Basic sequential pattern learning")
    print("2. Stacked RNN: Deeper architecture allows learning more complex patterns")
    print("3. Bidirectional RNN: Considers both past and future context")
    
    # Compare validation losses
    print("\nFinal Validation Losses:")
    print(f"Vanilla RNN: {vanilla_history.history['val_loss'][-1]:.4f}")
    print(f"Stacked RNN: {stacked_history.history['val_loss'][-1]:.4f}")
    print(f"Bidirectional RNN: {bidirectional_history.history['val_loss'][-1]:.4f}")

if __name__ == "__main__":
    main()
