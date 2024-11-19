# Text Generation with GPT-2 Fine-Tuning

This project demonstrates how to fine-tune a GPT-2 model for text generation tasks. The workflow involves preparing a custom dataset, fine-tuning the GPT-2 model using PyTorch and Hugging Face Transformers, and generating text with the fine-tuned model.

---
## Model Fine-Tuning

### Initialize the GPT-2 Model:
- Load the GPT-2 model and tokenizer from the Hugging Face Transformers library.

### Dataset Handling:
- Use a custom PyTorch dataset to tokenize the text file and prepare it for training.

### Training:
- Train the model using the AdamW optimizer for 3 epochs (or more based on dataset size and compute resources).

### Save the Model:
- Save the fine-tuned model for future use.

---

## Evaluation

### Loss Calculation:
- Evaluate the model on a test dataset to calculate the evaluation loss.

### Generate Example Text:
- Test the model by generating sample text for various prompts.

---

## Text Generation
Use the fine-tuned model to generate text based on custom prompts:

```python
prompt = "Once upon a time"
generated_text = generate_text(prompt)
print(f"Generated Text:\n{generated_text}")
