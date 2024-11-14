# Sentiment Classification using nlptown/bert-base-multilingual-uncased-sentiment

This repository demonstrates how to perform sentiment classification using the `nlptown/bert-base-multilingual-uncased-sentiment` model from Hugging Face's Transformers library. The model provides star ratings (from 0 to 4) and is used to classify sentences into **Positive**, **Neutral**, or **Negative** sentiments.

## Overview

The project performs sentiment analysis on a list of sentences and visualizes the distribution of sentiments. It uses the BERT-based model to assign star ratings to each sentence and then classifies the sentiment based on the number of stars:
- **Negative**: If the star rating is less than 3.
- **Neutral**: If the star rating is exactly 3.
- **Positive**: If the star rating is greater than 3.

The sentiment distribution is then plotted using `matplotlib`, with each sentiment represented by a specific color (Green for Positive, Red for Negative, Yellow for Neutral).

## Setup and Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/sentiment-classification.git
   cd sentiment-classification
Install the required dependencies:

bash
Copy code
pip install transformers matplotlib
Run the script sentiment_classification.py to perform sentiment analysis and generate the output image.

Code Explanation
The code uses Hugging Face's pipeline for sentiment analysis. It processes a list of sample sentences, classifies them into Positive, Neutral, or Negative sentiments, and then visualizes the sentiment distribution in a bar chart.

Steps:
Load the Pre-trained Model:
The model nlptown/bert-base-multilingual-uncased-sentiment is loaded using Hugging Face's pipeline.
Sentiment Classification:
Each sentence is classified, and the star rating is used to determine the sentiment (Negative, Neutral, or Positive).
Result Visualization:
The sentiment results are counted, and a bar chart is created showing the number of sentences for each sentiment.
Output:
The sentiment distribution bar chart is displayed using matplotlib.
Example Output
The output of the script will display the predicted sentiment for each sentence and show a bar chart representing the sentiment distribution.

Here is a sample output of the sentiment predictions:

vbnet
Copy code
Sentence: 'The website is easy to navigate, but the checkout process is slow.'
Predicted Sentiment: Neutral (Stars: 3, Confidence: 0.56)

Sentence: 'I love the fabric quality, but the stitching is poor.'
Predicted Sentiment: Neutral (Stars: 3, Confidence: 0.62)

Sentence: 'The customer service was very helpful, and the delivery was on time.'
Predicted Sentiment: Positive (Stars: 5, Confidence: 0.51)

Sentence: 'The room was clean, but the air conditioning was broken.'
Predicted Sentiment: Negative (Stars: 2, Confidence: 0.42)

Sentence: 'I didn’t like the product at all; it feels overpriced and flimsy.'
Predicted Sentiment: Negative (Stars: 2, Confidence: 0.50)

Sentence: 'The gym has modern equipment, but it’s often overcrowded.'
Predicted Sentiment: Neutral (Stars: 3, Confidence: 0.53)

Sentence: 'The laptop is lightweight and fast, but it heats up quickly.'
Predicted Sentiment: Neutral (Stars: 3, Confidence: 0.49)

Sentence: 'The meal was delicious, but the portions were too small.'
Predicted Sentiment: Neutral (Stars: 3, Confidence: 0.54)

Sentence: 'The movie was engaging, and the cinematography was stunning.'
Predicted Sentiment: Positive (Stars: 4, Confidence: 0.45)

Sentence: 'The software is slow, and it’s full of bugs.'
Predicted Sentiment: Negative (Stars: 2, Confidence: 0.46)
Output Image
The sentiment distribution will be visualized in the following bar chart:


In this chart:

Green represents Positive sentiment.
Red represents Negative sentiment.
Yellow represents Neutral sentiment.
License
This repository is licensed under the MIT License. See the LICENSE file for more details.

Acknowledgments
Hugging Face Transformers
PyTorch
Matplotlib
Feel free to explore the repository, modify the code, and experiment with other sentences or models!

vbnet
Copy code

### Instructions to Save the Output Image:
To save the generated bar chart (sentiment distribution), you can modify the code to save the plot as an image. Here's an update to the plotting part of the code:

```python
# Save the plot as an image
plt.savefig('output_image.png')
