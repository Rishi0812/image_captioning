import spacy
import csv
from sklearn.metrics.pairwise import cosine_similarity


def find_closest_sentences(input_sentence, sentences, top_k=5):
    nlp = spacy.load('en_core_web_md')

    # Calculate the vector representation of the input sentence
    input_vector = nlp(input_sentence).vector

    # Calculate the vector representation of each sentence in the list
    sentence_vectors = [nlp(sentence).vector for sentence in sentences]

    # Calculate cosine similarity between the input vector and sentence vectors
    similarities = cosine_similarity([input_vector], sentence_vectors)[0]

    # Sort the sentences based on similarity scores
    sorted_indexes = sorted(range(len(similarities)),
                            key=lambda i: similarities[i], reverse=True)

    # Get the top k similar sentences
    closest_sentences = [sentences[idx] for idx in sorted_indexes[:top_k]]

    return closest_sentences


# Read sentences from a CSV file
def read_sentences_from_csv(file_path, column_name):
    sentences = []
    with open(file_path, 'r', newline='') as file:
        reader = csv.DictReader(file)
        for row in reader:
            sentences.append(row[column_name])
    return sentences
