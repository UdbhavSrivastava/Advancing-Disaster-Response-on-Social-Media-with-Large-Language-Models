import re
import string
import emoji
from bs4 import BeautifulSoup
from autocorrect import Speller
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))
spell = Speller(lang='en')
abbreviations = {
    "thnx": "thanks", "thx": "thanks", "btw": "by the way", "pls": "please", "plz": "please",
    "u": "you", "r": "are", "ur": "your", "y": "why", "b4": "before", "gr8": "great",
    "imo": "in my opinion", "idk": "I don't know", "w8": "wait", "bday": "birthday"
}

def replace_emojis(text):
    return emoji.demojize(text, delimiters=(" ", " "))

def replace_abbreviations(text):
    words = text.split()
    return ' '.join([abbreviations.get(word, word) for word in words])

def clean_text(text):
    text = text.lower()
    text = BeautifulSoup(text, "lxml").get_text()
    text = replace_emojis(text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'rt\s+', '', text)
    text = re.sub(r'http\S+|www\S+', '', text)
    text = replace_abbreviations(text)
    text = spell(text)
    words = text.split()
    text = ' '.join(words)
    text = re.sub(r'\s+', ' ', text).strip()
    return text