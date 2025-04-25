import nltk
from nltk.corpus import treebank
from nltk.tag import hmm
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import time

# Download required NLTK data
nltk.download('treebank')

def load_and_preprocess_data():
    # Load tagged sentences from Penn Treebank
    tagged_sentences = treebank.tagged_sents()
    
    # Split into training and test sets
    train_sents, test_sents = train_test_split(tagged_sentences, test_size=0.2, random_state=42)
    return train_sents, test_sents

def viterbi_tagger(train_sents):
    # Train HMM tagger
    trainer = hmm.HiddenMarkovModelTrainer()
    tagger = trainer.train(train_sents)
    return tagger

def alternative_tagger(train_sents):
    # Use NLTK's built-in tagger as an alternative
    # This is a simplified version that doesn't actually train on the data
    # but demonstrates the concept
    tagger = nltk.tag.UnigramTagger(train_sents)
    return tagger

def evaluate_tagger(tagger, test_sents):
    # Evaluate tagger performance
    correct = 0
    total = 0
    
    start_time = time.time()
    for sent in test_sents:
        words = [word for word, tag in sent]
        actual_tags = [tag for word, tag in sent]
        predicted_tags = [tag for word, tag in tagger.tag(words)]
        
        correct += sum(1 for act, pred in zip(actual_tags, predicted_tags) if act == pred)
        total += len(actual_tags)
    end_time = time.time()
    
    accuracy = correct / total
    processing_time = end_time - start_time
    
    return accuracy, processing_time

def main():
    # Load and preprocess data
    print("Loading and preprocessing data...")
    train_sents, test_sents = load_and_preprocess_data()
    
    # Train and evaluate Viterbi HMM tagger
    print("\nTraining Viterbi HMM tagger...")
    viterbi_model = viterbi_tagger(train_sents)
    viterbi_accuracy, viterbi_time = evaluate_tagger(viterbi_model, test_sents)
    
    # Train and evaluate alternative tagger
    print("\nTraining Alternative tagger...")
    alt_model = alternative_tagger(train_sents)
    alt_accuracy, alt_time = evaluate_tagger(alt_model, test_sents)
    
    # Print results and comparison
    print("\nResults:")
    print(f"Viterbi HMM Tagger:")
    print(f"Accuracy: {viterbi_accuracy:.4f}")
    print(f"Processing time: {viterbi_time:.2f} seconds")
    
    print(f"\nAlternative Tagger:")
    print(f"Accuracy: {alt_accuracy:.4f}")
    print(f"Processing time: {alt_time:.2f} seconds")
    
    print("\nComparison Discussion:")
    print("1. Accuracy: ", end="")
    if viterbi_accuracy > alt_accuracy:
        print("Viterbi HMM performed better in terms of accuracy")
    else:
        print("Alternative tagger performed better in terms of accuracy")
    
    print("2. Computational Efficiency: ", end="")
    if viterbi_time < alt_time:
        print("Viterbi HMM was more computationally efficient")
    else:
        print("Alternative tagger was more computationally efficient")
    
    print("\n3. Generalization: The Alternative tagger uses a different approach")
    print("   that may capture different linguistic patterns compared to the HMM.")

if __name__ == "__main__":
    main()
