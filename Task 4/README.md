# T5 Text Summarization System

This is an interactive Python script that uses the **T5-large model** from the Hugging Face Transformers library to summarize input text. The system accepts user input, processes it with the T5 model, and generates a concise summary.

---

## **Features**
- Summarizes long paragraphs into concise summaries.
- Interactive prompt for real-time text summarization.
- Supports customization of summary length and quality.

---

## **Requirements**
- Python 3.8 or later
- Hugging Face Transformers library
- PyTorch

---

## **Installation**
1. Clone the repository or download the script file:
   ```bash
   git clone https://github.com/your-username/t5-summarization.git
   cd t5-summarization
   
## **Install the required libraries:**

pip install transformers torch

---
## **Steps:** ##
**Initialization:**
Loads the T5-large model and tokenizer from the Hugging Face Transformers library.

**Text Processing:**
Prepares the input with the "summarize:" prefix for the T5 model.
Tokenizes and truncates the text to meet model constraints.

**Summarization:**
Generates a summary using beam search for higher quality.

**Output:**
Decodes and displays the summarized text.

**Interactive Mode:**
Continuously accepts user input until the user decides to exit.

**Customization**
You can modify the summarization settings in the summarize_text function:
max_length: Maximum summary length.
min_length: Minimum summary length.
length_penalty: Adjusts the balance between concise and detailed summaries.
num_beams: Number of beams for beam search.
