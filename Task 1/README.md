Sentiment Analysis with nlptown/bert-base-multilingual-uncased-sentiment
This repository demonstrates how to perform sentiment analysis on a list of sentences using the pre-trained nlptown/bert-base-multilingual-uncased-sentiment model from Hugging Face's Transformers library. The model assigns star ratings (0-4) and classifies the sentiment of each sentence into one of three categories: Positive, Neutral, or Negative.

Overview
The project applies sentiment analysis to a set of sentences, classifying them based on the star ratings provided by the model. The classification is mapped to three sentiment categories:

Positive: Assigned to sentences with a star rating greater than 3.
Neutral: Assigned to sentences with a star rating of exactly 3.
Negative: Assigned to sentences with a star rating less than 3.
The sentiment distribution is visualized using a bar chart, with different colors representing each sentiment category:

Green for Positive
Yellow for Neutral
Red for Negative
Features
Star Rating Mapping: Sentences are analyzed and rated on a 0-4 star scale.
Sentiment Classification: Sentences are classified into Positive, Neutral, or Negative based on the star rating.
Visualization: The distribution of sentiment types is visualized using a bar chart.
Installation
Clone the Repository:

bash
Copy code
git clone https://github.com/your-username/sentiment-classification.git
cd sentiment-classification
Install the Required Dependencies:

bash
Copy code
pip install transformers matplotlib
Run the Script:
Execute the script sentiment_classification.py to perform sentiment analysis and generate the output image.

Usage
Code Walkthrough
1. Load the Pre-trained Model
The model nlptown/bert-base-multilingual-uncased-sentiment is loaded via Hugging Face’s pipeline for sentiment analysis.

2. Sentiment Analysis
The list of sentences is processed, and the model assigns star ratings to each sentence.
Based on the star rating, the sentiment is classified into one of three categories: Positive, Neutral, or Negative.

3. Visualization
The sentiment distribution is plotted in a bar chart format.
The chart shows the counts of Positive, Neutral, and Negative sentences.

Example Output
The following example output shows how the script classifies the sentences:

plaintext
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
Sentiment Distribution Visualization
The sentiment distribution for the sentences is visualized as a bar chart. The chart displays the number of sentences in each sentiment category (Positive, Neutral, Negative).

Here’s the generated output image:



In the chart:

Positive sentiment is represented in Green.
Neutral sentiment is represented in Yellow.
Negative sentiment is represented in Red.
Requirements
Python 3.6 or higher
Hugging Face Transformers library
Matplotlib library
