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


