# Health-Chatbot

## What is this?
This chatbot was developed by me in the span of approximately 2 months. This is my first try at natural langage processing and this extensive of a use of Python's **Sklearn** library.

## What should it do and how does it do it?
The chatbot was designed to perform three functions:

1. Correctly receive the user's name, no matter how complicated it is, as well as simply take in their birth date. This was achieved with the help of **RegEx**s.
2. Analyzing the user's input, the chatbot decided whether the user is healthy or not. This was done by training a **Logistic Regression** model on a word dataset usually associated with describing someone doing unwell. (**multi-layer perception** and **linear support vector** classifiers were also tried for this task but did not perform as well/predict pereived health of the user as accurately) A TF-IDF model trained on >300 documents from Google news was also utilized here.
3. Given the user's more casual/relaxed input, the chatbot analyzed their speech to try and predict their personality. (are they more person-focused, are they talkative, etc) This was achieved by running the user input throuhg a tagger that marked what part of speech each word within the input belonged to. The results were passed into a function that modified those tags to correspond with the **Penn Treebag** tags, which were finally used to decide what the user's speech communicates about their personality.


## What did that look like?
### Example of a conversation with a healthy person
![image](https://user-images.githubusercontent.com/87237231/154590055-ca39e5e0-a3ba-4b80-9213-cf07a753db1b.png)

### Example of a conversation with an unhealthy person
![image](https://user-images.githubusercontent.com/87237231/154590802-b1458898-5c81-42e3-9f59-e88d606bb212.png)
