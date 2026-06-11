import speech_recognition as sr
from pydub import AudioSegment
from textblob import TextBlob
import textwrap
import re


# Format transcribed text
def format_text(text, width=80):
    sentences = text.replace('. ', '.\n').replace('? ', '?\n').replace('! ', '!\n')
    return textwrap.fill(sentences, width=width)


# Lexical Diversity: Type-Token Ratio
def lexical_diversity(text):
    words = re.findall(r'\b\w+\b', text.lower())
    total_words = len(words)
    unique_words = len(set(words))
    score = unique_words / total_words if total_words else 0
    return score, total_words, unique_words


# Sentence Complexity: Avg sentence length and conjunctions
def sentence_complexity(text):
    sentences = re.split(r'[.!?]', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    num_sentences = len(sentences)
    words = re.findall(r'\b\w+\b', text)

    avg_sentence_length = len(words) / num_sentences if num_sentences else 0

    # Look for conjunctions and subordinating words
    conjunctions = ['and', 'but', 'or', 'because', 'although', 'since',
                    'while', 'if', 'when', 'though']
    conjunction_count = sum(text.lower().count(c) for c in conjunctions)

    if avg_sentence_length > 12 and conjunction_count >= num_sentences:
        feedback = "✅ Balanced complexity."
    else:
        feedback = "⚠️ Simple or flat sentence structure."

    return avg_sentence_length, conjunction_count, feedback


# File and language
audio_file = "/content/2024-02-24_21-25-20.WAV"
language = 'en-US'

recognizer = sr.Recognizer()

try:
    # Preprocess audio
    audio = AudioSegment.from_file(audio_file)
    audio = audio.set_frame_rate(16000)
    temp_wav_file = "temp.wav"
    audio.export(temp_wav_file, format="wav")

    with sr.AudioFile(temp_wav_file) as source:
        audio_data = recognizer.record(source)

    # Transcription
    text = recognizer.recognize_google(audio_data, language=language)
    formatted_text = format_text(text)

    # Sentiment
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    subjectivity = blob.sentiment.subjectivity

    # Lexical Diversity
    diversity, total_words, unique_words = lexical_diversity(text)
    if diversity > 0.5:
        lex_pattern = "✅ Diverse vocabulary."
    else:
        lex_pattern = "⚠️ Repeated words or less variety."

    # Sentence Complexity
    avg_len, conj_count, complexity_feedback = sentence_complexity(text)

    # Output
    print("\n--- Transcribed Text ---\n")
    print(formatted_text)

    print("\n--- Sentiment Score ---\n")
    print(f"Polarity Score    : {polarity:.2f} (−1 = negative, +1 = positive)")
    print(f"Subjectivity Score: {subjectivity:.2f} (0 = objective, 1 = subjective)")

    print("\n--- Lexical Diversity ---\n")
    print(f"Total Words       : {total_words}")
    print(f"Unique Words      : {unique_words}")
    print(f"Diversity Score   : {diversity:.2f} (Unique / Total)")
    print(f"Pattern Check     : {lex_pattern}")

    print("\n--- Sentence Complexity ---\n")
    print(f"Average Sentence Length : {avg_len:.2f} words")
    print(f"Conjunction Count        : {conj_count}")
    print(f"Complexity Feedback      : {complexity_feedback}")

except sr.UnknownValueError:
    print("Speech Recognition could not understand audio.")
except sr.RequestError as e:
    print(f"Could not request results from Speech Recognition service: {e}")
except Exception as e:
    print(f"An error occurred: {e}")
