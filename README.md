# Admission-Help-Desk-Chatbot-using-NLP
This code sets up a simple chatbot using Flask, which responds to user questions based on a set of pre-defined FAQs loaded from a JSON file. The chatbot can correct spelling mistakes, preprocess the text, and find the best match using cosine similarity on TF-IDF vectors
# FAQ Chatbot with Flask

This project is a simple FAQ chatbot built with Flask. The chatbot responds to user queries by matching them against a set of predefined FAQs using Natural Language Processing (NLP) techniques. The bot can handle spelling correction and context, particularly for queries related to department fees.

## Features

- **Spelling Correction:** The chatbot corrects spelling errors in user queries.
- **Question Matching:** The bot uses TF-IDF vectorization and cosine similarity to match user questions with FAQs.
- **Context Handling:** The bot can handle context, such as asking for additional information when a query is about fees.
- **Flask Web App:** The chatbot is served as a web application using Flask.

## Installation

### Prerequisites

- Python 3.x
- pip

### Setup

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/faq-chatbot.git
   cd faq-chatbot
Install the required Python packages:

bash
Copy code
pip install -r requirements.txt
