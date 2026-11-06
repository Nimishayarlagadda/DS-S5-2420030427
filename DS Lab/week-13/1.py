
import string
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

text = "The movie was amazing, exciting, and full of action scenes!"

text = text.lower()
text = text.translate(str.maketrans('', '', string.punctuation))

tokens = word_tokenize(text)

stop_words = set(stopwords.words('english'))

filtered_tokens = [
    word for word in tokens
    if word not in stop_words
]

print("Original Text:")
print("The movie was amazing, exciting, and full of action scenes!")

print("\nProcessed Tokens:")
print(filtered_tokens)
