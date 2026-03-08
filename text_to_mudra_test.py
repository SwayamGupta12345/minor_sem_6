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


# import pandas as pd
# import ast
# import re
# from deep_translator import GoogleTranslator
# import nltk
# from nltk.stem import WordNetLemmatizer
# import os
# # Download wordnet if not installed
# # nltk.download('wordnet')

# nltk.data.path.append("./nltk_data")
# # print("NLTK search paths:", nltk.data.path)
# lemmatizer = WordNetLemmatizer()

# # ===============================
# # LOAD DATASET
# # ===============================

# print("Loading mudra dataset...\n")

# df = pd.read_csv("./data/mudras.csv")

# print("Columns in dataset:")
# print(df.columns)

# # convert string lists → python lists
# df["meanings"] = df["meanings"].apply(ast.literal_eval)
# df["viniyoga"] = df["viniyoga"].apply(ast.literal_eval)

# print("\nDataset preview:")
# print(df.head())


# # ===============================
# # BUILD SEMANTIC DICTIONARY
# # meanings + viniyoga
# # ===============================

# mudra_dict = {}

# for _, row in df.iterrows():

#     mudra = row["transliteration"]

#     # meanings
#     for meaning in row["meanings"]:
#         mudra_dict[meaning.lower()] = mudra

#     # viniyoga phrases
#     for phrase in row["viniyoga"]:

#         words = re.findall(r'\b\w+\b', phrase.lower())

#         for w in words:
#             mudra_dict[w] = mudra


# print("\nMudra dictionary created")
# print("Total semantic mappings:", len(mudra_dict))


# # ===============================
# # TEXT PREPROCESSING
# # ===============================

# def preprocess(text):

#     text = text.lower()

#     words = re.findall(r'\b\w+\b', text)

#     # lemmatization
#     words = [lemmatizer.lemmatize(w) for w in words]

#     return words


# # ===============================
# # TRANSLATION (Hindi support)
# # ===============================

# def translate_if_needed(text):

#     try:
#         translated = GoogleTranslator(
#             source='auto', target='en').translate(text)
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
#     seen = set()

#     for word in words:

#         if word in mudra_dict:

#             m = mudra_dict[word]

#             if m not in seen:
#                 mudras.append(m)
#                 seen.add(m)

#     return translated, mudras


# # ===============================
# # BUILT-IN TEST SENTENCES
# # ===============================

# test_sentences = [

#     "The river flows in the forest",
#     "The king holds a crown",
#     "Fire rises in the night",
#     "A bird flies in the wind",
#     "The cloud covers the sky",
#     "A flower is offered to god",
#     "राजा के सिर पर मुकुट है",
#     "नदी जंगल में बहती है",
#     "आसमान में बादल हैं"

# ]


# print("\n==============================")
# print("Running built-in tests")
# print("==============================\n")

# for s in test_sentences:

#     translated, mudras = sentence_to_mudras(s)

#     print("Input:", s)
#     print("Translated:", translated)

#     if mudras:
#         print("Mudra sequence:", " → ".join(mudras))
#     else:
#         print("Mudra sequence: None found")

#     print("---------------------------------")


# # ===============================
# # INTERACTIVE MODE
# # ===============================

# print("\n==============================")
# print("Interactive Mudra Generator")
# print("==============================")

# while True:

#     user_input = input("\nEnter sentence (or type 'exit'): ")

#     if user_input.lower() == "exit":
#         break

#     translated, mudras = sentence_to_mudras(user_input)

#     print("\nEnglish interpretation:", translated)

#     if mudras:
#         print("Mudra sequence:", " → ".join(mudras))
#     else:
#         print("No matching mudras found")

# print("\nProgram finished.")


import pandas as pd
import ast
import re
from deep_translator import GoogleTranslator
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

nltk.data.path.append("./nltk_data")

lemmatizer = WordNetLemmatizer()

# ===============================
# LOAD DATASETS
# ===============================

print("Loading datasets...\n")

mudra_df = pd.read_csv("./data/mudras.csv")
face_df = pd.read_csv("./data/facial_expressions.csv")

# convert string lists → python lists
mudra_df["meanings"] = mudra_df["meanings"].apply(ast.literal_eval)
mudra_df["viniyoga"] = mudra_df["viniyoga"].apply(ast.literal_eval)

face_df["meanings"] = face_df["meanings"].apply(ast.literal_eval)
face_df["viniyoga"] = face_df["viniyoga"].apply(ast.literal_eval)

print("Mudra dataset loaded:", len(mudra_df))
print("Facial dataset loaded:", len(face_df))

emotion_normalization = {

    "happy": "happiness",
    "joyful": "joy",
    "sad": "sadness",
    "angry": "anger",
    "scared": "fear",
    "fearful": "fear",
    "afraid": "fear",
    "peaceful": "peace",
    "calm": "peace",
    "brave": "bravery",
    "disgusted": "disgust"

}
# ===============================
# BUILD SEMANTIC DICTIONARIES
# ===============================

mudra_dict = {}
face_dict = {}

# ---- Mudra dictionary ----
for _, row in mudra_df.iterrows():

    gesture = row["transliteration"]

    for meaning in row["meanings"]:
        mudra_dict[meaning.lower()] = gesture

    for phrase in row["viniyoga"]:
        words = re.findall(r'\b\w+\b', phrase.lower())
        for w in words:
            mudra_dict[w] = gesture


# ---- Facial expression dictionary ----
for _, row in face_df.iterrows():

    gesture = row["transliteration"]

    for meaning in row["meanings"]:

        meaning = meaning.lower()

        # normalize meaning
        lemma = lemmatizer.lemmatize(meaning)

        face_dict[meaning] = gesture
        face_dict[lemma] = gesture

    # process viniyoga phrases
    for phrase in row["viniyoga"]:

        words = re.findall(r'\b\w+\b', phrase.lower())

        for w in words:

            lemma = lemmatizer.lemmatize(w)

            face_dict[w] = gesture
            face_dict[lemma] = gesture
print("\nMudra semantic mappings:", len(mudra_dict))
print("Facial semantic mappings:", len(face_dict))


# ===============================
# TEXT PREPROCESSING
# ===============================

def preprocess(text):

    text = text.lower()

    words = re.findall(r'\b\w+\b', text)

    processed = []

    for w in words:

        lemma_n = lemmatizer.lemmatize(w, pos='n')
        lemma_v = lemmatizer.lemmatize(w, pos='v')
        lemma_a = lemmatizer.lemmatize(w, pos='a')

        processed.extend([w, lemma_n, lemma_v, lemma_a])

    return list(set(processed))


# ===============================
# TRANSLATION
# ===============================

def translate_if_needed(text):

    try:
        translated = GoogleTranslator(
            source='auto', target='en').translate(text)
        return translated
    except:
        return text


# ===============================
# SENTENCE → GESTURES
# ===============================

def sentence_to_gestures(sentence):

    translated = translate_if_needed(sentence)
    words = preprocess(translated)

    mudras = []
    expressions = []

    seen_mudra = set()
    seen_face = set()

    for word in words:

        # normalize emotion words
        if word in emotion_normalization:
            word = emotion_normalization[word]

        if word in mudra_dict:

            m = mudra_dict[word]

            if m not in seen_mudra:
                mudras.append(m)
                seen_mudra.add(m)

        if word in face_dict:

            f = face_dict[word]

            if f not in seen_face:
                expressions.append(f)
                seen_face.add(f)
    return translated, mudras, expressions


# ===============================
# TEST SENTENCES
# ===============================

# test_sentences = [

#     "The river flows in the forest",
#     "The king holds a crown",
#     "A flower is offered with love",
#     "The warrior shows anger",
#     "The child laughs with joy",
#     "राजा के सिर पर मुकुट है",
#     "नदी जंगल में बहती है"

# ]
test_sentences = [

    # Mudra focused
    "The river flows in the forest",
    "The king holds a crown",
    "A flower is offered to god",

    # Emotion + mudra
    "A flower is offered with love",
    "The warrior shows anger",
    "The child laughs with joy",

    # Facial expressions
    "The girl feels happy",
    "The boy is joyful",
    "The woman is sad",
    "The man becomes angry",
    "The child is scared",
    "The devotee feels peaceful",
    "The hero stands brave",
    "The person feels disgust",

    # Hindi tests
    "राजा के सिर पर मुकुट है",
    "नदी जंगल में बहती है",
    "बच्चा खुश है",
    "वह डर गया",
    "वह गुस्से में है"
]


print("\n==============================")
print("Running built-in tests")
print("==============================\n")

for s in test_sentences:

    translated, mudras, expressions = sentence_to_gestures(s)

    print("Input:", s)
    print("Translated:", translated)

    if mudras:
        print("Mudras:", " → ".join(mudras))
    else:
        print("Mudras: None")

    if expressions:
        print("Facial Expression:", " → ".join(expressions))
    else:
        print("Facial Expression: None")

    print("---------------------------------")


# ===============================
# INTERACTIVE MODE
# ===============================

print("\n==============================")
print("Dance Gesture Generator")
print("==============================")

while True:

    user_input = input("\nEnter sentence (or type 'exit'): ")

    if user_input.lower() == "exit":
        break

    translated, mudras, expressions = sentence_to_gestures(user_input)

    print("\nEnglish interpretation:", translated)

    if mudras:
        print("Mudras:", " → ".join(mudras))
    else:
        print("Mudras: None found")

    if expressions:
        print("Facial Expressions:", " → ".join(expressions))
    else:
        print("Facial Expressions: None found")

print("\nProgram finished.")
