import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, Conv1D, MaxPooling1D, GlobalMaxPooling1D
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
import matplotlib.pyplot as plt

# Constants
MAX_FEATURES = 10000  # Maximum number of words to consider
MAX_LEN = 500  # Maximum length of input sequences
EMBEDDING_DIM = 100  # Dimension of word embeddings
BATCH_SIZE = 32
EPOCHS = 10

def load_and_preprocess_data():
    """
    Load and preprocess the IMDB dataset
    """
    # Load the IMDB dataset
    (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=MAX_FEATURES)
    
    # Pad sequences
    x_train = pad_sequences(x_train, maxlen=MAX_LEN)
    x_test = pad_sequences(x_test, maxlen=MAX_LEN)
    
    return (x_train, y_train), (x_test, y_test)

def create_cnn_model(filter_sizes=[3, 4, 5], num_filters=128):
    """
    Create a 1D CNN model with multiple filter sizes
    """
    model = Sequential([
        # Embedding layer
        Embedding(MAX_FEATURES, EMBEDDING_DIM, input_length=MAX_LEN),
        
        # Convolutional layers with different filter sizes
        *[Conv1D(num_filters, filter_size, activation='relu') for filter_size in filter_sizes],
        *[MaxPooling1D(pool_size=2) for _ in filter_sizes],
        
        # Global max pooling
        GlobalMaxPooling1D(),
        
        # Dense layers
        Dense(128, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    
    return model

def train_and_evaluate_model(model, x_train, y_train, x_test, y_test):
    """
    Train and evaluate the CNN model
    """
    # Compile the model
    model.compile(optimizer='adam',
                 loss='binary_crossentropy',
                 metrics=['accuracy'])
    
    # Early stopping callback
    early_stopping = EarlyStopping(monitor='val_loss', patience=2)
    
    # Train the model
    history = model.fit(x_train, y_train,
                       batch_size=BATCH_SIZE,
                       epochs=EPOCHS,
                       validation_split=0.2,
                       callbacks=[early_stopping])
    
    # Evaluate the model
    test_loss, test_accuracy = model.evaluate(x_test, y_test)
    print(f"\nTest accuracy: {test_accuracy:.4f}")
    print(f"Test loss: {test_loss:.4f}")
    
    return history

def plot_training_history(history):
    """
    Plot training history
    """
    plt.figure(figsize=(12, 4))
    
    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.close()

def main():
    # Load and preprocess data
    print("Loading and preprocessing data...")
    (x_train, y_train), (x_test, y_test) = load_and_preprocess_data()
    
    # Create and train model with default filter sizes
    print("\nTraining model with default filter sizes [3, 4, 5]...")
    model_default = create_cnn_model()
    history_default = train_and_evaluate_model(model_default, x_train, y_train, x_test, y_test)
    plot_training_history(history_default)
    
    # Create and train model with different filter sizes
    print("\nTraining model with different filter sizes [2, 3, 4]...")
    model_alt = create_cnn_model(filter_sizes=[2, 3, 4])
    history_alt = train_and_evaluate_model(model_alt, x_train, y_train, x_test, y_test)
    plot_training_history(history_alt)

if __name__ == "__main__":
    main() 