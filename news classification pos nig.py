import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional
from sklearn.model_selection import train_test_split

# Load data
def load_imdb_data(data_dir):
    texts = []
    labels = []

    for label_type in ['neg', 'pos']:
        dir_name = os.path.join(data_dir, label_type)
        for fname in os.listdir(dir_name):
            if fname.endswith('.txt'):
                with open(os.path.join(dir_name, fname), encoding='utf-8') as f:
                    texts.append(f.read())
                labels.append(0 if label_type == 'neg' else 1)

    return texts, labels

# Specify the paths to the training and testing data directories
train_dir = "C:/datasets/aclImdb/train"
test_dir = "C:/datasets/aclImdb/test"

train_texts, train_labels = load_imdb_data(train_dir)
test_texts, test_labels = load_imdb_data(test_dir)

# Convert to numpy arrays
train_labels = np.array(train_labels)
test_labels = np.array(test_labels)

# Tokenize and pad sequences
vocab_size = 10000
max_length = 120
trunc_type = 'post'
oov_tok = "<OOV>"

tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(train_texts)
word_index = tokenizer.word_index

train_sequences = tokenizer.texts_to_sequences(train_texts)
train_padded = pad_sequences(train_sequences, maxlen=max_length, truncating=trunc_type)

test_sequences = tokenizer.texts_to_sequences(test_texts)
test_padded = pad_sequences(test_sequences, maxlen=max_length)

model = Sequential([
    Embedding(vocab_size, 64, input_length=max_length),
    Bidirectional(LSTM(64, return_sequences=True)),
    Bidirectional(LSTM(32)),
    Dense(24, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

num_epochs = 10
history = model.fit(train_padded, train_labels, epochs=num_epochs, validation_data=(test_padded, test_labels), verbose=2)

loss, accuracy = model.evaluate(test_padded, test_labels)
print(f'Loss: {loss}, Accuracy: {accuracy}')

# Define new reviews for prediction
new_reviews = [
    "The movie was fantastic! I really enjoyed the plot and the acting was superb.",
    "What a waste of time. The story was boring and the characters were not believable.",
    "An average movie. Some parts were good, but overall it was nothing special.",
    "Absolutely loved it! The best movie I've seen this year.",
    "Terrible. I walked out of the theater halfway through."
]

# Tokenize and pad new sequences
new_sequences = tokenizer.texts_to_sequences(new_reviews)
new_padded_sequences = pad_sequences(new_sequences, maxlen=max_length, truncating=trunc_type)

# Predict the sentiment of new reviews
predictions = model.predict(new_padded_sequences)

# Convert predictions to binary labels
predicted_labels = (predictions > 0.5).astype(int)

# Map predicted labels to sentiment
label_map = {0: 'Negative', 1: 'Positive'}
predicted_sentiments = [label_map[label[0]] for label in predicted_labels]

# Print the results
for review, sentiment in zip(new_reviews, predicted_sentiments):
    print(f'Review: "{review}" - Sentiment: {sentiment}')
