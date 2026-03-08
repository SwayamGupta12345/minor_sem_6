# import pandas as pd
# import ast
# import re
# from deep_translator import GoogleTranslator


# # ===============================
# # LOAD DATASET
# # ===============================

# print("Loading mudra dataset...")

# df = pd.read_csv("./data/mudras.csv")

# print("\nColumns in dataset:")
# print(df.columns)


# # convert list strings to python lists
# df["meanings"] = df["meanings"].apply(ast.literal_eval)
# df["viniyoga"] = df["viniyoga"].apply(ast.literal_eval)

# print("\nDataset preview:")
# print(df.head())


# # ===============================
# # CREATE MEANING → MUDRA MAP
# # ===============================

# mudra_dict = {}

# for _, row in df.iterrows():

#     mudra_name = row["transliteration"]

#     for meaning in row["meanings"]:
#         mudra_dict[meaning.lower()] = mudra_name


# print("\nMudra dictionary created")
# print("Total mappings:", len(mudra_dict))


# # ===============================
# # TEXT PREPROCESSING
# # ===============================

# def preprocess(text):

#     text = text.lower()
#     words = re.findall(r'\b\w+\b', text)

#     return words


# # ===============================
# # TRANSLATION (HINDI → ENGLISH)
# # ===============================

# def translate_if_needed(text):

#     try:
#         translated = GoogleTranslator(source='auto', target='en').translate(text)
#         return translated
#     except:
#         return text


# # ===============================
# # SENTENCE → MUDRA SEQUENCE
# # ===============================

# def sentence_to_mudras(sentence):

#     translated = translate_if_needed(sentence)

#     words = preprocess(translated)

#     mudras = []

#     for word in words:

#         if word in mudra_dict:
#             mudras.append(mudra_dict[word])

#     return translated, mudras


# # ===============================
# # BUILT IN TEST SENTENCES
# # ===============================

# test_sentences = [

#     "The river flows in the forest",
#     "The king holds a crown",
#     "Fire rises in the night",
#     "A bird flies in the wind",
#     "The cloud covers the sky",
#     "राजा के सिर पर मुकुट है",
#     "नदी जंगल में बहती है"

# ]


# print("\n==========================")
# print("Running built-in tests")
# print("==========================\n")

# for s in test_sentences:

#     translated, mudras = sentence_to_mudras(s)

#     print("Input:", s)
#     print("Translated:", translated)
#     print("Mudras:", mudras)
#     print("-----------------------")


# # ===============================
# # USER INPUT LOOP
# # ===============================

# print("\n==========================")
# print("Interactive Mudra Generator")
# print("==========================")

# while True:

#     user_input = input("\nEnter sentence (or type exit): ")

#     if user_input.lower() == "exit":
#         break

#     translated, mudras = sentence_to_mudras(user_input)

#     print("\nEnglish interpretation:", translated)

#     if mudras:
#         print("Mudra sequence:", " → ".join(mudras))
#     else:
#         print("No mudras found")

# print("\nProgram finished.")


import pandas as pd
import ast
import re
from deep_translator import GoogleTranslator
import nltk
from nltk.stem import WordNetLemmatizer
import os
# Download wordnet if not installed
# nltk.download('wordnet')

nltk.data.path.append("./nltk_data")
# print("NLTK search paths:", nltk.data.path)
lemmatizer = WordNetLemmatizer()

# ===============================
# LOAD DATASET
# ===============================

print("Loading mudra dataset...\n")

df = pd.read_csv("./data/mudras.csv")

print("Columns in dataset:")
print(df.columns)

# convert string lists → python lists
df["meanings"] = df["meanings"].apply(ast.literal_eval)
df["viniyoga"] = df["viniyoga"].apply(ast.literal_eval)

print("\nDataset preview:")
print(df.head())


# ===============================
# BUILD SEMANTIC DICTIONARY
# meanings + viniyoga
# ===============================

mudra_dict = {}

for _, row in df.iterrows():

    mudra = row["transliteration"]

    # meanings
    for meaning in row["meanings"]:
        mudra_dict[meaning.lower()] = mudra

    # viniyoga phrases
    for phrase in row["viniyoga"]:

        words = re.findall(r'\b\w+\b', phrase.lower())

        for w in words:
            mudra_dict[w] = mudra


print("\nMudra dictionary created")
print("Total semantic mappings:", len(mudra_dict))


# ===============================
# TEXT PREPROCESSING
# ===============================

def preprocess(text):

    text = text.lower()

    words = re.findall(r'\b\w+\b', text)

    # lemmatization
    words = [lemmatizer.lemmatize(w) for w in words]

    return words


# ===============================
# TRANSLATION (Hindi support)
# ===============================

def translate_if_needed(text):

    try:
        translated = GoogleTranslator(
            source='auto', target='en').translate(text)
        return translated
    except:
        return text


# ===============================
# SENTENCE → MUDRA SEQUENCE
# ===============================

def sentence_to_mudras(sentence):

    translated = translate_if_needed(sentence)

    words = preprocess(translated)

    mudras = []
    seen = set()

    for word in words:

        if word in mudra_dict:

            m = mudra_dict[word]

            if m not in seen:
                mudras.append(m)
                seen.add(m)

    return translated, mudras


# ===============================
# BUILT-IN TEST SENTENCES
# ===============================

test_sentences = [

    "The river flows in the forest",
    "The king holds a crown",
    "Fire rises in the night",
    "A bird flies in the wind",
    "The cloud covers the sky",
    "A flower is offered to god",
    "राजा के सिर पर मुकुट है",
    "नदी जंगल में बहती है",
    "आसमान में बादल हैं"

]


print("\n==============================")
print("Running built-in tests")
print("==============================\n")

for s in test_sentences:

    translated, mudras = sentence_to_mudras(s)

    print("Input:", s)
    print("Translated:", translated)

    if mudras:
        print("Mudra sequence:", " → ".join(mudras))
    else:
        print("Mudra sequence: None found")

    print("---------------------------------")


# ===============================
# INTERACTIVE MODE
# ===============================

print("\n==============================")
print("Interactive Mudra Generator")
print("==============================")

while True:

    user_input = input("\nEnter sentence (or type 'exit'): ")

    if user_input.lower() == "exit":
        break

    translated, mudras = sentence_to_mudras(user_input)

    print("\nEnglish interpretation:", translated)

    if mudras:
        print("Mudra sequence:", " → ".join(mudras))
    else:
        print("No matching mudras found")

print("\nProgram finished.")
