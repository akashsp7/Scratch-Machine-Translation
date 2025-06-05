# English to Spanish Neural Machine Translation using Bidirectional LSTM

# Author: Akash Pawar

# This project implements a sequence-to-sequence neural machine translation model that translates English sentences to Spanish using a Bidirectional LSTM encoder and LSTM decoder architecture.

# ## Project Overview

# This neural machine translation system:
# 1. Processes bilingual text data from English-Spanish parallel corpus
# 2. Implements a Bidirectional LSTM encoder for better context understanding
# 3. Uses standard LSTM decoder for target language generation
# 4. Incorporates multinomial sampling with temperature for diverse translations
# 5. Evaluates translation quality using BLEU scores

# ## Key Features

# - **Bidirectional LSTM Encoder**: Captures both forward and backward context for improved encoding
# - **Character-level Tokenization**: Fine-grained text processing for robust translation
# - **Temperature-based Sampling**: Balanced approach between diversity and accuracy
# - **BLEU Score Evaluation**: Quantitative assessment of translation quality

# 1. Data Preparation and Text Processing

# Loading and preprocessing the Spanish-English parallel corpus with comprehensive text cleaning.


import re
import string
from unicodedata import normalize
import numpy

# load doc into memory
def load_doc(filename):
    # open the file as read only
    file = open(filename, mode='rt', encoding='utf-8')
    # read all text
    text = file.read()
    # close the file
    file.close()
    return text


# split a loaded document into sentences
def to_pairs(doc):
    lines = doc.strip().split('\n')
    pairs = [line.split('\t') for line in  lines]
    return pairs

def clean_data(lines):
    cleaned = list()
    # prepare regex for char filtering
    re_print = re.compile('[^%s]' % re.escape(string.printable))
    # prepare translation table for removing punctuation
    table = str.maketrans('', '', string.punctuation)
    for pair in lines:
        clean_pair = list()
        for line in pair:
            # normalize unicode characters
            line = normalize('NFD', line).encode('ascii', 'ignore')
            line = line.decode('UTF-8')
            # tokenize on white space
            line = line.split()
            # convert to lowercase
            line = [word.lower() for word in line]
            # remove punctuation from each token
            line = [word.translate(table) for word in line]
            # remove non-printable chars form each token
            line = [re_print.sub('', w) for w in line]
            # remove tokens with numbers in them
            line = [word for word in line if word.isalpha()]
            # store as string
            clean_pair.append(' '.join(line))
        cleaned.append(clean_pair)
    return numpy.array(cleaned)


# Dataset configuration
filename = 'Data/spa.txt'  # Spanish-English parallel corpus
n_train = 20000  # Number of training samples

# load dataset
doc = load_doc(filename)

# split into English-Spanish pairs
pairs = to_pairs(doc)

# clean sentences
clean_pairs = clean_data(pairs)[0:n_train, :]

# Display sample translations
for i in range(3000, 3010):
    print('[' + clean_pairs[i, 0] + '] => [' + clean_pairs[i, 1] + ']')


input_texts = clean_pairs[:, 0]
target_texts = ['\t' + text + '\n' for text in clean_pairs[:, 1]]

print('Length of input_texts:  ' + str(input_texts.shape))
print('Length of target_texts: ' + str(input_texts.shape))

max_encoder_seq_length = max(len(line) for line in input_texts)
max_decoder_seq_length = max(len(line) for line in target_texts)

print('max length of input  sentences: %d' % (max_encoder_seq_length))
print('max length of target sentences: %d' % (max_decoder_seq_length))


# 2. Text Preprocessing and Tokenization

# 2.1. Character-level Sequence Conversion

# Converting text data to numerical sequences with character-level tokenization for fine-grained translation control.


from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# encode and pad sequences
def text2sequences(max_len, lines):
    tokenizer = Tokenizer(char_level=True, filters='')
    tokenizer.fit_on_texts(lines)
    seqs = tokenizer.texts_to_sequences(lines)
    seqs_pad = pad_sequences(seqs, maxlen=max_len, padding='post')
    return seqs_pad, tokenizer.word_index, tokenizer


encoder_input_seq, input_token_index, encode_tokenizer = text2sequences(max_encoder_seq_length,
                                                      input_texts)
decoder_input_seq, target_token_index, _ = text2sequences(max_decoder_seq_length,
                                                       target_texts)

print('shape of encoder_input_seq: ' + str(encoder_input_seq.shape))
print('shape of input_token_index: ' + str(len(input_token_index)))
print('shape of decoder_input_seq: ' + str(decoder_input_seq.shape))
print('shape of target_token_index: ' + str(len(target_token_index)))

num_encoder_tokens = len(input_token_index) + 1
num_decoder_tokens = len(target_token_index) + 1

print('num_encoder_tokens: ' + str(num_encoder_tokens))
print('num_decoder_tokens: ' + str(num_decoder_tokens))


# 2.2. One-hot Encoding

# Converting tokenized sequences to one-hot encoded tensors for neural network processing.

from tensorflow.keras.utils import to_categorical

# one hot encode target sequence
def onehot_encode(sequences, max_len, vocab_size):
    n = len(sequences)
    data = numpy.zeros((n, max_len, vocab_size))
    for i in range(n):
        data[i, :, :] = to_categorical(sequences[i], num_classes=vocab_size)
    return data

encoder_input_data = onehot_encode(encoder_input_seq, max_encoder_seq_length, num_encoder_tokens)
decoder_input_data = onehot_encode(decoder_input_seq, max_decoder_seq_length, num_decoder_tokens)

decoder_target_seq = numpy.zeros(decoder_input_seq.shape)
decoder_target_seq[:, 0:-1] = decoder_input_seq[:, 1:]
decoder_target_data = onehot_encode(decoder_target_seq,
                                    max_decoder_seq_length,
                                    num_decoder_tokens)

print(encoder_input_data.shape)
print(decoder_input_data.shape)

# 3. Neural Network Architecture
 
# Building a sophisticated seq2seq model with Bidirectional LSTM encoder and standard LSTM decoder.

# 3.1. Bidirectional LSTM Encoder
 
# The encoder processes English input and creates contextual representations using bidirectional processing.

from tensorflow.keras.layers import Input, LSTM
from tensorflow.keras.models import Model
from keras.layers import Bidirectional, Concatenate

latent_dim = 256

# inputs of the encoder network
encoder_inputs = Input(shape=(None, num_encoder_tokens),
                       name='encoder_inputs')

# Bidirectional LSTM for enhanced context understanding
encoder_bilstm = Bidirectional(LSTM(latent_dim, return_state=True,
                    dropout=0.5, name='encoder_bilstm'))
_, forward_h, forward_c, backward_h, backward_c = encoder_bilstm(encoder_inputs)

# Concatenate forward and backward states
state_h = Concatenate()([forward_h, backward_h])
state_c = Concatenate()([forward_c, backward_c])

# build the encoder network model
encoder_model = Model(inputs=encoder_inputs,
                      outputs=[state_h, state_c],
                      name='encoder')


# Model visualization and summary

from IPython.display import SVG
from keras.utils import model_to_dot, plot_model

SVG(model_to_dot(encoder_model, show_shapes=False).create(prog='dot', format='svg'))

plot_model(
    model=encoder_model, show_shapes=False,
    to_file='encoder.pdf'
)

encoder_model.summary()


# 3.2. LSTM Decoder Network

# The decoder generates Spanish translations using the encoder's contextual states.

from tensorflow.keras.layers import Input, LSTM, Dense
from tensorflow.keras.models import Model

# inputs of the decoder network
decoder_input_h = Input(shape=(latent_dim*2,), name='decoder_input_h')
decoder_input_c = Input(shape=(latent_dim*2,), name='decoder_input_c')
decoder_input_x = Input(shape=(None, num_decoder_tokens), name='decoder_input_x')

# LSTM layer with doubled dimension to match concatenated encoder states
decoder_lstm = LSTM(latent_dim*2, return_sequences=True,
                    return_state=True, dropout=0.5, name='decoder_lstm')
decoder_lstm_outputs, state_h, state_c = decoder_lstm(decoder_input_x,
                                                      initial_state=[decoder_input_h, decoder_input_c])

# Dense layer for character probability distribution
decoder_dense = Dense(num_decoder_tokens, activation='softmax', name='decoder_dense')
decoder_outputs = decoder_dense(decoder_lstm_outputs)

# build the decoder network model
decoder_model = Model(inputs=[decoder_input_x, decoder_input_h, decoder_input_c],
                      outputs=[decoder_outputs, state_h, state_c],
                      name='decoder')

from IPython.display import SVG
from keras.utils import model_to_dot, plot_model

SVG(model_to_dot(decoder_model, show_shapes=False).create(prog='dot', format='svg'))

plot_model(
    model=decoder_model, show_shapes=False,
    to_file='decoder.pdf'
)

decoder_model.summary()


# 3.3. Complete Seq2Seq Architecture

# Connecting encoder and decoder for end-to-end training.

# input layers
encoder_input_x = Input(shape=(None, num_encoder_tokens), name='encoder_input_x')
decoder_input_x = Input(shape=(None, num_decoder_tokens), name='decoder_input_x')

# connect encoder to decoder
encoder_final_states = encoder_model([encoder_input_x])
decoder_lstm_output, _, _ = decoder_lstm(decoder_input_x, initial_state=encoder_final_states)
decoder_pred = decoder_dense(decoder_lstm_output)

model = Model(inputs=[encoder_input_x, decoder_input_x],
              outputs=decoder_pred,
              name='model_training')

from IPython.display import SVG
from keras.utils import model_to_dot, plot_model

SVG(model_to_dot(model, show_shapes=False).create(prog='dot', format='svg'))

plot_model(
    model=model, show_shapes=False,
    to_file='model_training.pdf'
)

model.summary()


# 3.4. Model Training

# Training the complete seq2seq model on the bilingual dataset.


print('shape of encoder_input_data' + str(encoder_input_data.shape))
print('shape of decoder_input_data' + str(decoder_input_data.shape))
print('shape of decoder_target_data' + str(decoder_target_data.shape))

model.compile(optimizer='rmsprop', loss='categorical_crossentropy')

model.fit([encoder_input_data, decoder_input_data],  
          decoder_target_data,                       
          batch_size=64, epochs=50, validation_split=0.2)

model.save('seq2seq.h5')


# ## 4. Translation Generation

# 4.1. Inference Pipeline
 
# Implementing the translation process with temperature-controlled sampling for diverse outputs.

# Reverse-lookup token index to decode sequences back to readable text
reverse_input_char_index = dict((i, char) for char, i in input_token_index.items())
reverse_target_char_index = dict((i, char) for char, i in target_token_index.items())


def decode_sequence(input_seq, temperature=0.1):
    """
    Generate Spanish translation from English input using multinomial sampling.
    
    Args:
        input_seq: Encoded English sentence
        temperature: Controls randomness (lower = more focused, higher = more diverse)
    
    Returns:
        Translated Spanish sentence
    """
    states_value = encoder_model.predict(input_seq, verbose = 0)

    target_seq = numpy.zeros((1, 1, num_decoder_tokens))
    target_seq[0, 0, target_token_index['\t']] = 1.

    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value, verbose=0)

        # Multinomial sampling with temperature for diverse translations
        output_tokens = output_tokens[0][0].astype('float64')
        softmax_probs = numpy.exp(numpy.log(output_tokens)/temperature) / numpy.sum(numpy.exp(numpy.log(output_tokens)/temperature))
        multinomial_sample = numpy.random.multinomial(1, softmax_probs)
        sampled_token_index = numpy.argsort(multinomial_sample)[-1]
        if sampled_token_index == 0:
            sampled_token_index = numpy.argsort(multinomial_sample)[-2]

        sampled_char = reverse_target_char_index[sampled_token_index]
        decoded_sentence += sampled_char

        if (sampled_char == '\n' or
           len(decoded_sentence) > max_decoder_seq_length):
            stop_condition = True

        target_seq = numpy.zeros((1, 1, num_decoder_tokens))
        target_seq[0, 0, sampled_token_index] = 1.

        states_value = [h, c]

    return decoded_sentence

# Sample translations from training data
for seq_index in range(2100, 2120):
    input_seq = encoder_input_data[seq_index: seq_index + 1]
    decoded_sentence = decode_sequence(input_seq, temperature=0.1)
    print('-')
    print('English:       ', input_texts[seq_index])
    print('Spanish (true): ', target_texts[seq_index][1:-1])
    print('Spanish (pred): ', decoded_sentence[0:-1])


# 4.2. Custom Translation Function
 
# Translating arbitrary English sentences to Spanish.

def translate_sentence(input_sentence, temperature=0.1):
    """
    Translate any English sentence to Spanish.
    
    Args:
        input_sentence: English text to translate
        temperature: Sampling temperature for diversity
    
    Returns:
        Spanish translation
    """
    input_sentence = numpy.array([input_sentence])
    input_sequence = pad_sequences(encode_tokenizer.texts_to_sequences(input_sentence), 
                                   maxlen=max_encoder_seq_length, padding='post')
    input_x = onehot_encode(input_sequence, max_encoder_seq_length, num_encoder_tokens)
    translated_sentence = decode_sequence(input_x, temperature=temperature)
    
    return translated_sentence[:-1]  # Remove end token

# Example translation
input_sentence = 'I love you'
translated = translate_sentence(input_sentence)
print('English: ' + input_sentence)
print('Spanish: ' + translated)


# 5. Model Evaluation using BLEU Score

# Comprehensive evaluation of translation quality using BLEU metrics on held-out test data.

# 5.1. Enhanced Dataset Partitioning
 
# Expanding dataset and creating proper train/validation/test splits for robust evaluation.

# Expanded dataset for better evaluation
n_train = 40000
doc = load_doc(filename)
pairs = to_pairs(doc)
clean_pairs = clean_data(pairs)[0:n_train, :]
input_texts = clean_pairs[:, 0]
target_texts = ['\t' + text + '\n' for text in clean_pairs[:, 1]]

print('Length of input_texts:  ' + str(input_texts.shape))
print('Length of target_texts: ' + str(len(target_texts)))

from sklearn.model_selection import train_test_split

# Create train/test split
input_texts, input_texts_test, target_texts, target_texts_test = train_test_split(
    input_texts, target_texts, test_size=0.1, random_state=42)

max_encoder_seq_length = max(len(line) for line in input_texts)
max_decoder_seq_length = max(len(line) for line in target_texts)

print('max length of input  sentences: %d' % (max_encoder_seq_length))
print('max length of target sentences: %d' % (max_decoder_seq_length))

# Rebuild tokenizers on training data only
encoder_input_seq, input_token_index, encoder_tokenizer = text2sequences(max_encoder_seq_length,
                                                      input_texts)
decoder_input_seq, target_token_index, _ = text2sequences(max_decoder_seq_length,
                                                       target_texts)

print('shape of encoder_input_seq: ' + str(encoder_input_seq.shape))
print('shape of input_token_index: ' + str(len(input_token_index)))
print('shape of decoder_input_seq: ' + str(decoder_input_seq.shape))
print('shape of target_token_index: ' + str(len(target_token_index)))

num_encoder_tokens = len(input_token_index) + 1
num_decoder_tokens = len(target_token_index) + 1

print('num_encoder_tokens: ' + str(num_encoder_tokens))
print('num_decoder_tokens: ' + str(num_decoder_tokens))

# Prepare training data
encoder_input_data = onehot_encode(encoder_input_seq, max_encoder_seq_length, num_encoder_tokens)
decoder_input_data = onehot_encode(decoder_input_seq, max_decoder_seq_length, num_decoder_tokens)

decoder_target_seq = numpy.zeros(decoder_input_seq.shape)
decoder_target_seq[:, 0:-1] = decoder_input_seq[:, 1:]
decoder_target_data = onehot_encode(decoder_target_seq,
                                    max_decoder_seq_length,
                                    num_decoder_tokens)

print(encoder_input_data.shape)
print(decoder_input_data.shape)

# Create train/validation split
encoder_input_data, encoder_input_data_val, decoder_input_data, decoder_input_data_val, decoder_target_data, decoder_target_data_val = train_test_split(
    encoder_input_data, decoder_input_data, decoder_target_data, test_size=0.2, random_state=42)

print(encoder_input_data.shape)
print(decoder_input_data.shape)

print(encoder_input_data_val.shape)
print(decoder_input_data_val.shape)


# 5.2 Enhanced Model Training with Optimization

# Training the Bidirectional LSTM model with advanced optimization techniques.

# Rebuild model architecture
latent_dim = 256

encoder_inputs = Input(shape=(None, num_encoder_tokens), name='encoder_inputs')

encoder_bilstm = Bidirectional(LSTM(latent_dim, return_state=True,
                    dropout=0.5, name='encoder_bilstm'))
_, forward_h, forward_c, backward_h, backward_c = encoder_bilstm(encoder_inputs)

state_h = Concatenate()([forward_h, backward_h])
state_c = Concatenate()([forward_c, backward_c])

encoder_model = Model(inputs=encoder_inputs,
                      outputs=[state_h, state_c],
                      name='encoder')

# Decoder with proper dimensions
decoder_input_h = Input(shape=(512,), name='decoder_input_h')
decoder_input_c = Input(shape=(512,), name='decoder_input_c')
decoder_input_x = Input(shape=(None, num_decoder_tokens), name='decoder_input_x')

decoder_lstm = LSTM(latent_dim*2, return_sequences=True,
                    return_state=True, dropout=0.5, name='decoder_lstm')
decoder_lstm_outputs, state_h, state_c = decoder_lstm(decoder_input_x,
                                                      initial_state=[decoder_input_h, decoder_input_c])

decoder_dense = Dense(num_decoder_tokens, activation='softmax', name='decoder_dense')
decoder_outputs = decoder_dense(decoder_lstm_outputs)

decoder_model = Model(inputs=[decoder_input_x, decoder_input_h, decoder_input_c],
                      outputs=[decoder_outputs, state_h, state_c],
                      name='decoder')

# Complete model
encoder_input_x = Input(shape=(None, num_encoder_tokens), name='encoder_input_x')
decoder_input_x = Input(shape=(None, num_decoder_tokens), name='decoder_input_x')

encoder_final_states = encoder_model([encoder_input_x])
decoder_lstm_output, _, _ = decoder_lstm(decoder_input_x, initial_state=encoder_final_states)
decoder_pred = decoder_dense(decoder_lstm_output)

model = Model(inputs=[encoder_input_x, decoder_input_x],
              outputs=decoder_pred,
              name='model_training')

from tensorflow.keras.optimizers.legacy import Adam
from keras.callbacks import LearningRateScheduler

def lr_scheduler(epoch, lr):
    """Learning rate scheduler for adaptive training."""
    if epoch < 30:
        return lr
    else:
        if epoch % 5 == 0:
            return lr * 0.9
        else:
            return lr
    
optimizer = Adam(learning_rate=0.001)

model.compile(optimizer=optimizer, loss='categorical_crossentropy')

learning_rate_scheduler = LearningRateScheduler(lr_scheduler)

# Train with validation data and learning rate scheduling
model.fit([encoder_input_data, decoder_input_data],  
          decoder_target_data,                       
          batch_size=64, 
          epochs=50, 
          validation_data=([encoder_input_data_val, decoder_input_data_val], decoder_target_data_val),
          callbacks=[learning_rate_scheduler])

model.save('seq2seq_enhanced.h5')

# Update reverse lookup dictionaries
reverse_input_char_index = dict((i, char) for char, i in input_token_index.items())
reverse_target_char_index = dict((i, char) for char, i in target_token_index.items())

# Test enhanced model
input_sentence = 'I love you'
input_sentence = numpy.array([input_sentence])
input_sequence = pad_sequences(encoder_tokenizer.texts_to_sequences(input_sentence), 
                               maxlen=max_encoder_seq_length, padding='post')
input_x = onehot_encode(input_sequence, max_encoder_seq_length, num_encoder_tokens)
translated_sentence = decode_sequence(input_x, temperature=0.5)

print('English: ' + input_sentence[0])
print('Spanish: ' + translated_sentence[:-1])


# 5.3 BLEU Score Evaluation

# Quantitative evaluation of translation quality using BLEU metrics on the test set.

from nltk.translate.bleu_score import sentence_bleu

# Comprehensive BLEU evaluation
total_score = 0
num_samples = len(input_texts_test)

print("Evaluating translation quality using BLEU score...")
print("=" * 50)

for i in range(num_samples):
    input_sentence = input_texts_test[i]
    input_sentence = numpy.array([input_sentence])
    input_sequence = pad_sequences(encoder_tokenizer.texts_to_sequences(input_sentence), 
                                   maxlen=max_encoder_seq_length, padding='post')
    input_x = onehot_encode(input_sequence, max_encoder_seq_length, num_encoder_tokens)
    translated_sentence = decode_sequence(input_x)

    # Calculate BLEU score for this translation
    bleu_score = sentence_bleu([target_texts_test[i][1:-1]], translated_sentence[0:-1], weights=[1])
    total_score += bleu_score
    
    # Progress reporting
    if (i+1) % 1000 == 0 or i == 0:
        avg_score = total_score / (i+1)
        print(f'Processed {i+1}/{num_samples} samples: Average BLEU = {avg_score:.4f}')

final_bleu_score = total_score / num_samples
print("=" * 50)
print(f'Final BLEU Score on Test Set: {final_bleu_score:.4f}')
print("=" * 50)

# Performance interpretation
if final_bleu_score > 0.3:
    print("🎉 Excellent translation quality!")
elif final_bleu_score > 0.2:
    print("✅ Good translation quality")
elif final_bleu_score > 0.1:
    print("⚡ Reasonable translation quality")
else:
    print("🔄 Translation quality needs improvement")
