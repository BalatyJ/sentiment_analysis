# This file implements a Naive Bayes Classifier
import math


class BayesClassifier():
    """
    Naive Bayes Classifier
    file length: file length of training file
    sections: sections for incremental training
    """

    def __init__(self):
        self.positive_word_counts = {}
        self.negative_word_counts = {}

        self.total_positive_sentences = 0
        self.percent_positive_sentences = 0
        self.total_negative_sentences = 0
        self.percent_negative_sentences = 0

        self.file_length = 499
        self.file_sections = [self.file_length // 4,
                              self.file_length // 3, self.file_length // 2]

    def train(self, train_data, train_labels, vocab, start_increment, end_increment):
        """
        This function builds the word counts and sentence percentages used for classify_text
        train_data: vectorized text
        train_labels: vectorized labels
        vocab: vocab from build_vocab
        """

        # Initialize values for negative and positive word counts if the dicts are empty.
        if (len(self.negative_word_counts) == 0 or len(self.positive_word_counts) == 0):
            for word in vocab:
                self.negative_word_counts[word] = 0
                self.positive_word_counts[word] = 0

        end_increment = min(self.file_length, end_increment)

        for i in range(start_increment, end_increment):

            # Increment training labels.
            if (train_labels[i] == 0):
                self.total_negative_sentences += 1
            else:
                self.total_positive_sentences += 1

            for j in range(len(train_data[i])):
                # Indicates word is present in the document.
                if (train_data[i][j] == 1):

                    # If class label is 0, we add 1 to the negative word counts.
                    if (train_labels[i] == 0):
                        self.negative_word_counts[vocab[j]] += 1

                    # Otherwise we add 1 to the positive word counts.
                    else:
                        self.positive_word_counts[vocab[j]] += 1

        return 1

    def classify_text(self, vectors, vocab):
        """
        vectors: [vector1, vector2, ...]
        predictions: [0, 1, ...]
        """

        # We find the probability of a word being positive by dividing the number of
        # occurrences of a word being positive by all of the positive words.
        # A similar calculation is done to find the probability of a word
        # being negative.

        # The probability of the classifier is the number of occurrences of a
        # sentence being positive or negative and dividing that by the sum of the
        # number for both positive and negative sentences.

        predictions = []

        self.percent_positive_sentences = self.total_positive_sentences / \
            (self.total_negative_sentences + self.total_positive_sentences)

        self.percent_negative_sentences = 1 - self.percent_positive_sentences

        # For each text vector, we determine using Bayes' Algorithm whether it's more
        # likely to be positive or negative.

        # Find the probability for the vector being negative and positive, and pick the max value.
        # Already accounts for P(classlabel = 0).

        for vector in vectors:
            # To construct the normalizing factor, we need to find the probability of each word in the vector.
            prob_vector_negative = math.log(
                self.percent_negative_sentences, 2)
            prob_vector_positive = math.log(
                self.percent_positive_sentences, 2)

            for idx in range(len(vector)):
                # We use the index in the vector to calculate
                # the conditional probabilities for each sentence
                word = vocab[idx]

                if (vector[idx] == 1):
                    # If the vector is equal to 1, we include
                    # it in the calculation of our probabilities.

                    prob_vector_negative += math.log((self.negative_word_counts[word] + 1)/(
                        self.total_negative_sentences + len(vocab)), 2)

                    prob_vector_positive += math.log((self.positive_word_counts[word] + 1)/(
                        self.total_positive_sentences + len(vocab)), 2)

                elif (vector[idx] == 0):
                    # We find the complement of the probability
                    # of the word not being in the sentence.

                    prob_vector_negative += math.log(1/(len(vocab)), 2)
                    prob_vector_positive += math.log(1/(len(vocab)), 2)

            # Then we take the max of the two values.
            predict_class_label = 0 if prob_vector_positive < prob_vector_negative else 1
            predictions.append(predict_class_label)

        return predictions
