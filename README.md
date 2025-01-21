#  Arabic POS Tagger
In natural language processing (NLP), **Part of Speech (POS)** refers to the **grammatical category** or syntactic function that a word serves in a sentence. It's a way of **categorizing words** based on their roles within the structure of a sentence. POS tagging involves assigning a specific label, such as `noun`, `verb`, `adjective`, `adverb`, etc., to each word in a sentence.

Here are some common parts of speech:
| Tag              | Arabic Tag | Description |
| :---------------- | ------: | :---- |
| Noun (N)        |   اسم   | Represents a person, place, thing, or idea. Examples: dog, city, happiness. |
| Verb (V)           |   فعل   | Describes an action or occurrence. Examples: run, eat, sleep. |
| Adjective (ADJ)    |  صفة   | Modifies or describes a noun. Examples: happy, tall, red. |
| Adverb (ADV) |  حال   | Modifies or describes a verb, adjective, or other adverb. Examples: quickly, very, well. |
| Pronoun (PRON) |  ضمير   | Replaces a noun. Examples: he, she, it. |
| Preposition (PREP) |  حرف جر   | Indicates relationships between words, often in terms of time or place. Examples: in, on, under. |
| Conjunction (CONJ) |  اقتران   | Connects words, phrases, or clauses. Examples: and, but, or. |
| Interjection (INTJ) |  تعجب   | Expresses strong emotion. Examples: wow, oh, ouch. |

**POS tagging** is an essential task in NLP because understanding the grammatical structure of a sentence helps machines **comprehend the meaning and context of the text**. It's particularly useful in applications like **text analysis**, **information retrieval**, and **language translation**.

# Tools Used
The project is implemented using the following Python packages:

| Package | Description |
| --- | --- |
| NumPy | Numerical computing library |
| Pandas | Data manipulation library |
| Matplotlib | Data visualization library |
| Sklearn | Machine learning library |
| pytorch | Open-source machine learning framework |
| Transformers | Hugging Face package contains state-of-the-art Natural Language Processing models |
| Datasets | Hugging Face package contains datasets |
| seqeval | Metric used to evaluate tokens   |

# Dataset
The [Arabic-PADT UD treebank](https://github.com/UniversalDependencies/UD_Arabic-PADT) is based on the Prague Arabic Dependency Treebank (PADT), created at the Charles University in Prague.

The treebank consists of **7,664** sentences (282,384 tokens) and its domain is mainly **newswire**. The annotation is licensed under the terms of CC BY-NC-SA 3.0 and its original (non-UD) version can be downloaded from http://hdl.handle.net/11858/00-097C-0000-0001-4872-3.

The morphological and syntactic annotation of the Arabic UD treebank is created through the conversion of PADT data. The conversion procedure has been designed by Dan Zeman. The main coordinator of the original PADT project was Otakar Smrž.

| Split | Number of Samples |
| --- | --- |
| train | 6075 |
| validation | 909 |
| test | 680 |

# Methodology
## Dataset Preparation

This document outlines the methodology and steps taken to investigate and preprocess the dataset for Part-of-Speech (POS) tagging.

## Dataset Description
The dataset consists of sentences with associated POS tags. Each row represents a word or token from a sentence, along with its corresponding POS tag. The columns in the dataset include:
- **text**: The word or token.
- **tags**: The POS tag associated with the word.

## Steps for Dataset Investigation and Preprocessing

### 1. Adding Sentence ID Column
To differentiate between sentences, a `sentence_id` column was added to the dataset. This ensures that each sentence is uniquely identified, making it easier to process and analyze individual sentences.

### 2. Processing Sentences
Each sentence was processed to be treated as a single input unit rather than individual words. This step ensures that the context of the sentence is preserved for POS tagging.

### 3. Finding Unique POS Tags
The unique labels for each POS tag were identified. This step is crucial for understanding the variety of tags in the dataset and preparing for model training.

### 4. Feature Selection
Only the relevant columns for POS tagging were selected. The selected columns are:
- `sentence_id`: Identifier for the sentence.
- `id`: Identifier for the word within the sentence.
- `form`: The word or token.
- `upos`: The universal POS tag.

These columns were mapped to a standardized format using the following mapping:
```python
column_mapper = {
    "SENTENCE_ID": "sentence_id",
    "ID": "id",
    "FORM": "word",
    "UPOS": "tag",
}

```
# Model Selection for Arabic POS Tagging

For effective POS tagging in Arabic, selecting an appropriate tokenizer is crucial. Using a tokenizer trained on an Arabic corpus ensures proper tokenization, avoiding issues such as subtoken misalignment between labels (tags) and candidate words.

## Tokenizer Selection

We evaluated two tokenizers for their suitability in handling Arabic text:

### 1. **XLM-RoBERTa Tokenizer**
   - **Model**: [`FacebookAI/xlm-roberta-base`](https://huggingface.co/FacebookAI/xlm-roberta-base)
   - **Description**: XLM-RoBERTa is a multilingual model trained on a large corpus of approximately **100 languages**, including Arabic. This makes it highly effective for tokenizing Arabic text.

### 2. **BERT Tokenizer**
   - **Model**: [`BERT-base-uncased`](https://huggingface.co/google-bert/bert-base-uncased)
   - **Description**: BERT is a widely used model, but its tokenizer is primarily trained on English text. When applied to Arabic, it often breaks words into subtokens on each char, leading to misalignment between tokens and their corresponding labels.

## Comparison of Tokenizers

After examining the tokenization outputs of both models:
- **XLM-RoBERTa** produces **dominant tokens** for Arabic text, preserving the integrity of words.
- **BERT** often generates **char subtokens** for Arabic words, causing misalignment issues.

### Example:
- **Arabic Word**: "اللغة العربية"
  - **XLM-RoBERTa Tokenizer**: `['<s>', '▁رفض', 'ت', '▁بر', 'لين', '▁ان', '▁ت', 'رفع', '▁الح', 'صار', '▁من', '▁على', '▁اي', 'طال', 'يا', '</s>']`

  - **BERT Tokenizer**: `['[CLS]', 'ر', '##ف', '##ض', '##ت', 'ب', '##ر', '##ل', '##ي', '##ن', 'ا', '##ن', 'ت', '##ر', '##ف', '##ع', 'ا', '##ل', '##ح', '##ص', '##ا', '##ر', 'م', '##ن', 'ع', '##ل', '##ى', 'ا', '##ي', '##ط', '##ا', '##ل', '##ي', '##ا', '[SEP]']`

## Conclusion
### XLM-RoBERTa: Multilingual Transformer Model

#### Overview
**XLM-RoBERTa** is a state-of-the-art multilingual transformer-based model developed by Facebook AI. It is optimized for cross-lingual tasks, handling over 100 languages effectively, including low-resource ones. This model builds upon the RoBERTa architecture and employs Masked Language Modeling (MLM) for pretraining, similar to BERT.

---

#### Model Details

| **Aspect**                   | **XLM-RoBERTa**                                                                                                         |
|-------------------------------|-------------------------------------------------------------------------------------------------------------------------|
| **Full Form**                | Cross-Lingual Language Model RoBERTa                                                                                    |
| **Developer**                | Facebook AI Research                                                                                                    |
| **Architecture**             | Optimized RoBERTa architecture                                                                                         |
| **Pretraining Objective**    | Masked Language Modeling (MLM): Predicts randomly masked tokens in sentences based on their context.                     |
| **Languages Covered**        | 100 languages, including high-resource and low-resource languages like Swahili and Maltese.                             |
| **Pretraining Dataset**      | CommonCrawl dataset (2.5TB of filtered text data).                                                                      |
| **Tokenizer**                | SentencePiece tokenizer with a vocabulary size of 250,000 subwords.                                                     |
| **Model Types**              | - `xlm-roberta-base` (125M parameters)                                                                                  |
|                               | - `xlm-roberta-large` (355M parameters)                                                                                 |
| **Strengths**                | - Handles multilingual tasks effectively.                                                                               |
|                               | - Supports low-resource languages.                                                                                      |
|                               | - Outperforms mBERT on cross-lingual benchmarks.                                                                        |
| **Weaknesses**               | - Large size requires more computational resources.                                                                      |
|                               | - Domain-specific tasks may need significant fine-tuning.                                                              |
| **Libraries/Frameworks**     | Hugging Face's `transformers`, PyTorch, TensorFlow.                                                                      |
| **Pretrained Model Size**    | - Base: ~1.1GB                                                                                                           |
|                               | - Large: ~4GB                                                                                                           |

---

#### Comparison: XLM-RoBERTa vs BERT (Base)

| **Aspect**                | **XLM-RoBERTa**                                                                                             | **BERT (Base)**                                                              |
|----------------------------|-------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------|
| **Developer**             | Facebook AI Research                                                                                      | Google AI                                                                   |
| **Architecture**          | Based on RoBERTa (optimized BERT).                                                                        | Original BERT architecture.                                                 |
| **Pretraining Objective** | Masked Language Modeling (MLM).                                                                            | Masked Language Modeling (MLM) and Next Sentence Prediction (NSP).          |
| **Languages Covered**     | 100 languages (multilingual).                                                                              | BERT (Base) is monolingual, trained only on English.                        |
| **Tokenizer**             | SentencePiece tokenizer (vocab size: 250,000).                                                             | WordPiece tokenizer (vocab size: 30,000).                                   |
| **Pretraining Dataset**   | 2.5TB of multilingual CommonCrawl data.                                                                    | 16GB of English data from BooksCorpus and Wikipedia.                        |
| **Model Parameters**      | - `Base`: 125M                                                                                            | - `Base`: 110M                                                              |
|                           | - `Large`: 355M                                                                                           | - `Large`: 340M                                                             |
| **Strengths**             | - Multilingual zero-shot and few-shot transfer.                                                           | - Highly effective for monolingual English tasks.                           |
|                           | - Handles low-resource languages well.                                                                    | - Simple and effective for English NLP tasks.                               |
| **Weaknesses**            | - Computationally expensive due to its size.                                                              | - Cannot handle cross-lingual tasks without separate training.              |
|                           | - Tokenizer has a large vocabulary, increasing complexity.                                                | - Limited to English, requiring separate models for other languages.        |
| **Performance**           | Outperforms mBERT and BERT on cross-lingual tasks (e.g., XNLI, MLQA).                                      | Excels in English monolingual tasks but lacks multilingual capabilities.     |

---

### Use Cases

#### XLM-RoBERTa
- Multilingual NLP applications.
- Translation tasks.
- Cross-lingual classification and question answering.
- Named Entity Recognition (NER) for multiple languages.

#### BERT (Base)
- English-specific NLP tasks.
- Sentiment analysis, question answering, and NER for English.
- Domain-specific fine-tuning with English corpora.

---

## Resources
- [Hugging Face: xlm-roberta-base](https://huggingface.co/xlm-roberta-base)
- [Hugging Face: xlm-roberta-large](https://huggingface.co/xlm-roberta-large)
- [Hugging Face: bert-base-uncased](https://huggingface.co/bert-base-uncased)
- [Original XLM-RoBERTa Paper](https://arxiv.org/abs/1911.02116)
- [Original BERT Paper](https://arxiv.org/abs/1810.04805)


The **XLM-RoBERTa tokenizer** is better suited for Arabic POS tagging due to its multilingual training, which includes Arabic.Therefore, we recommend using the `xlmr_tokenizer` for processing Arabic text in this project.


## Use the Model as a Hugging Face pipeline:
```python
from transformers import pipeline

pos_tagger = pipeline("token-classification", "Mohamedsheded33/xlm-roberta-base-finetuned-ud-arabic")
text = "اشترى خالد سيارة، و أصبح يمتلك 3 سيارات."

predictions = pos_tagger(text)
words = [item["word"] for item in predictions]
predicted_entities = [item["entity"] for item in predictions]


print(f"words:    {words}")
print(f"entites:  {predicted_entities}")
```

**output**:
```
words:    ['▁اشتر', 'ى', '▁خالد', '▁سيارة', '،', '▁و', '▁أصبح', '▁يمتلك', '▁3', '▁سيارات', '.']
entites:  ['VERB', 'VERB', 'X', 'NOUN', 'PUNCT', 'CCONJ', 'VERB', 'VERB', 'NUM', 'NOUN', 'PUNCT']
```



# Results
Here is the table of results after fine-tuning for 3 epochs:
| Epoch              | Training Loss | Validation Loss | F1 | Accuracy |
| ---------------- | ------ | ---- |  ---- |  ---- |
| 1 | 0.188700 | 0.113953 | 0.958816 | 0.971454 |
| 2 | 0.090000 | 0.090725 | 0.966489 | 0.976796 |
| 3 | 0.055800 | 0.091719 | 0.969978 | 0.979393 |

# Testing
## Test dataset metrics
| Metric                   | Value              |
|--------------------------|--------------------|
| Test Loss               | 0.1091            |
| Test F1 Score           | 0.9639            |
| Test Accuracy           | 0.9745            |
| Test Runtime (seconds)  | 5.6217            |
| Test Samples/Second     | 120.961           |
| Test Steps/Second       | 60.48             |

## example
| Word        | Entity | Score    |
|-------------|--------|----------|
| —           | VERB   | 0.999519 |
| ذهب         | VERB   | 0.999655 |
| الطبيب       | NOUN   | 0.979260 |
| —           | ADP    | 0.999777 |
| —           | ADP    | 0.979798 |
| لـ          | ADP    | 0.999731 |
| المستشفى     | NOUN   | 0.989932 |
| .           | PUNCT  | 0.999757 |


# NetworkX Graph Representation
To visualize the transitions between POS tags, we can create a directed graph using NetworkX. Each unique POS tag is represented as a node, and edges represent transitions from one tag to another.
![image](https://github.com/user-attachments/assets/2f4434ec-64af-40f1-ac76-7bf8c844e597)

# Challenges and Solutions in Arabic NLP Project

This document outlines the primary challenges faced during our Arabic NLP project and the solutions implemented to address them.

---

## 1. Challenge: Finding a Pretrained Tokenizer for Arabic Corpus/Data

### Problem
Most tokenizers available are designed for English or multilingual corpora, which are not well-suited to the unique linguistic and morphological characteristics of the Arabic language. This created difficulties in accurately tokenizing Arabic text.

### Solution
- **Testing Multiple Tokenizers:** We evaluated several tokenizers, including those specifically pretrained on Arabic datasets (e.g., AraBERT) and multilingual models (e.g., mBERT, XLM-R).
- **Selection Process:** Through systematic evaluation based on tokenization quality and downstream task performance, we identified the most effective tokenizer for our project.

---

## 2. Challenge: Unalignment of Sub-Tokens with Labels

### Problem
When using subword tokenization methods like WordPiece or Byte-Pair Encoding, a single word may be split into multiple sub-tokens. This misalignment caused inconsistencies between sub-tokens and their respective labels, especially for sequence labeling tasks like Named Entity Recognition (NER).

### Solution
- **Alignment Function:** We developed an alignment function to handle this issue effectively:
  - **Ignoring Special Tokens:** Special tokens (e.g., `[CLS]`, `[SEP]`) and sub-tokens without corresponding labels were excluded.
  - **Setting Ignored Tokens to –100:** Sub-tokens without associated labels were assigned a label of `-100` to ensure they were ignored by the loss function during training.
  - **Maintaining Consistency:** This approach preserved the alignment between tokens and labels, leading to more reliable training and evaluation.

---

### Summary
By systematically addressing these challenges, we ensured robust handling of Arabic text, enabling the successful implementation of our NLP pipeline. These solutions improved both the quality of tokenization and the alignment necessary for effective sequence labeling tasks.



# References
- **XLM-RoBERTa Tokenizer**: [FacebookAI/xlm-roberta-base](https://huggingface.co/FacebookAI/xlm-roberta-base)
- **Seqeval Documentation**: [Seqeval Docs](https://github.com/chakki-works/seqeval)
- **Arabic POS Tagging with Machine Learning**: [Omdena Blog](https://www.omdena.com/blog/machine-learning-and-nlp-for-arabic-part-of-speech-tagging)
- **NER Fine-Tuning Tutorial**: [YouTube Tutorial](https://www.youtube.com/watch?v=Q1i4bIIFOFc)
