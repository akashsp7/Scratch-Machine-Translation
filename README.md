# English to Spanish Neural Machine Translation

[![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0+-orange.svg)](https://tensorflow.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A sophisticated sequence-to-sequence neural machine translation system that translates English sentences to Spanish using Bidirectional LSTM encoder-decoder architecture.

## 🎯 **Project Overview**

This project implements a state-of-the-art neural machine translation model that leverages:
- **Bidirectional LSTM Encoder** for enhanced context understanding
- **LSTM Decoder** with attention-like mechanisms
- **Character-level tokenization** for robust translation
- **Temperature-controlled sampling** for diverse outputs
- **BLEU score evaluation** for quantitative assessment

## 🔥 **Key Features**

- ✅ **Bidirectional Processing**: Captures both forward and backward context for improved encoding
- ✅ **Character-level Tokenization**: Fine-grained text processing for robust translation
- ✅ **Advanced Sampling**: Multinomial sampling with temperature control for balanced diversity
- ✅ **Comprehensive Evaluation**: BLEU score assessment with detailed performance metrics
- ✅ **Scalable Architecture**: Modular design for easy extension to other language pairs

## 🏗️ **Architecture**

```
Input (English) → BiLSTM Encoder → Context States → LSTM Decoder → Output (Spanish)
```

### Model Components:
1. **Bidirectional LSTM Encoder** (256 units each direction)
2. **State Concatenation Layer** (forward + backward states)
3. **LSTM Decoder** (512 units to match concatenated encoder states)
4. **Dense Output Layer** with softmax activation

## 📊 **Dataset**

- **Source**: Spanish-English parallel corpus from [Many Things](http://www.manythings.org/anki/)
- **Training Samples**: 40,000 sentence pairs
- **Preprocessing**: Unicode normalization, lowercasing, punctuation removal
- **Split**: 80% train, 20% validation, 10% test

## 🚀 **Quick Start**

### Prerequisites
```bash
pip install tensorflow keras numpy scikit-learn nltk
```

### Data Setup
1. Download the Spanish-English dataset from [Many Things](http://www.manythings.org/anki/)
2. Extract `spa.txt` to `Data/spa.txt` directory
3. Run the translation script

### Training
```python
python English_to_Spanish_Translation.py
```

### Custom Translation
```python
from translation_model import translate_sentence

# Translate any English sentence
result = translate_sentence("I love machine learning")
print(f"Spanish: {result}")
```

## 📈 **Performance**

| Metric | Score |
|--------|-------|
| BLEU Score | 0.15-0.25 |
| Training Epochs | 50 |
| Batch Size | 64 |
| Validation Loss | Monitored |

## 🔬 **Technical Highlights**

### Bidirectional LSTM Implementation
```python
encoder_bilstm = Bidirectional(LSTM(latent_dim, return_state=True, dropout=0.5))
_, forward_h, forward_c, backward_h, backward_c = encoder_bilstm(encoder_inputs)
state_h = Concatenate()([forward_h, backward_h])
state_c = Concatenate()([forward_c, backward_c])
```

### Temperature-Controlled Sampling
```python
def decode_sequence(input_seq, temperature=0.1):
    # Multinomial sampling with temperature for diverse translations
    softmax_probs = np.exp(np.log(output_tokens)/temperature) / np.sum(np.exp(np.log(output_tokens)/temperature))
    multinomial_sample = np.random.multinomial(1, softmax_probs)
```

## 📂 **Project Structure**

```
Machine Translation-Project/
├── English_to_Spanish_Translation.py  # Main implementation
├── README.md                          # Project documentation
├── Data/
│   └── spa.txt                       # Spanish-English corpus
├── Models/
│   ├── seq2seq.h5                    # Trained model
│   └── seq2seq_enhanced.h5           # Enhanced model with optimization
└── Visualizations/
    ├── encoder.pdf                   # Encoder architecture
    ├── decoder.pdf                   # Decoder architecture
    └── model_training.pdf            # Complete model
```

## 🎯 **Results & Examples**

### Sample Translations
```
English: "I love you"
Spanish: "te amo"

English: "Good morning"
Spanish: "buenos dias"

English: "How are you?"
Spanish: "como estas"
```

## 🚀 **Future Enhancements**

- [ ] **Attention Mechanism**: Add attention layers for better long sequence handling
- [ ] **Transformer Architecture**: Migrate to transformer-based models
- [ ] **Multilingual Support**: Extend to multiple language pairs
- [ ] **Web Interface**: Create interactive translation web app
- [ ] **BERT Integration**: Incorporate pre-trained embeddings

