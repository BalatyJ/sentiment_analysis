# CS331 Sentiment Analysis Assignment 3
# This file contains the processing functions
from classifier import *
from copy import *
import matplotlib.pyplot as plt


def process_text(text):
    """
    Preprocesses the text: Remove apostrophes, punctuation marks, etc.
    Returns a list of text
    """
    preprocessed_text = text

    remove_marks = ["'", '"', '!', '?', ":", ";",
                    ".", ",", "-", "_", "[", "]",
                    "{", "}", "(", ")", "2",
                    "3", "4", "5", "6", "7", "8",
                    "9", "*", "$", "%", "#", "&",
                    "@", "#", "+", "-", "/", "0", "1"]

    class_labels = []

    for index in range(len(preprocessed_text)):
        preprocessed_text[index] = preprocessed_text[index].lower()

        if (preprocessed_text[index].endswith("1 \n")):
            preprocessed_text[index] = preprocessed_text[index].rstrip(
                " 1\t\n")
            class_labels.append(1)
        else:
            preprocessed_text[index] = preprocessed_text[index].rstrip(
                " 0\t\n")
            class_labels.append(0)

        for mark in remove_marks:
            preprocessed_text[index] = preprocessed_text[index].replace(
                mark, "")

    return preprocessed_text, class_labels


def build_vocab(preprocessed_text, vocab):
    """
    Builds the vocab from the preprocessed text
    preprocessed_text: output from process_text
    Returns unique text tokens
    """

    for document in preprocessed_text:
        document = document.split()
        for word in document:
            wordAddFlag = 1
            for idx in range(len(vocab)):
                if (word == vocab[idx]):
                    wordAddFlag = 0
                    break

            if (wordAddFlag):
                vocab.append(word)

    vocab.sort()

    return vocab


def vectorize_text(text, vocab):
    """
    Converts the text into vectors
    text: preprocess_text from process_text
    vocab: vocab from build_vocab
    Returns the vectorized text and the labels
    """

    vectorized_text = []

    for document in text:

        sentence_vector = []
        split_document = document.split()
        # Iterate over all words in vocab and document.
        for i in range(len(vocab)):
            add0Flag = 1
            for j in range(len(split_document)):
                if (vocab[i] == split_document[j]):
                    sentence_vector.append(1)
                    add0Flag = 0
                    break
            if (add0Flag):
                sentence_vector.append(0)

        vectorized_text.append(copy(sentence_vector))

    return vectorized_text


def accuracy(predicted_labels, true_labels):
    """
    predicted_labels: list of 0/1s predicted by classifier
    true_labels: list of 0/1s from text file
    return the accuracy of the predictions
    """
    base = len(predicted_labels)
    correct = 0
    for idx in range(len(predicted_labels)):
        correct += 1 if predicted_labels[idx] == true_labels[idx] else 0
    accuracy_score = correct/base*100

    return accuracy_score


def listToString(list):
    text_result = ""
    for idx in range(len(list) - 1):
        text_result += f'{list[idx]},'
    text_result += f'{list[len(list) - 1]}'

    return text_result


def preprocessing_functionality():
    # vocab is the vocabulary for all test and training data.
    vocab = []

    with open("./trainingSet.txt", 'r') as sample_data:
        training_data = sample_data.readlines()

    with open("./testSet.txt", 'r') as sample_data:
        test_data = sample_data.readlines()

    preprocessed_training_data, training_labels = process_text(training_data)
    preprocessed_test_data, test_labels = process_text(test_data)

    # Then we continue building our vocab.
    vocab = build_vocab(preprocessed_training_data, vocab)

    # And get our vectorized text and corresponding labels.
    vectorized_training_data = vectorize_text(
        preprocessed_training_data, vocab)
    vectorized_test_data = vectorize_text(
        preprocessed_test_data, vocab)

    # Then we output the labels and vectorized text to files.
    with open("./preprocessed_train.txt", 'w') as output_data:
        labels_text = listToString(vocab)
        labels_text += ",classlabel\n"
        output_data.write(labels_text)
        for idx in range(len(vectorized_training_data)):
            vector_string = listToString(vectorized_training_data[idx])
            vector_string += f',{training_labels[idx]}\n'
            output_data.write(vector_string)

    with open("./preprocessed_test.txt", 'w') as output_data:
        labels_text = listToString(vocab)
        labels_text += ",classlabel\n"
        output_data.write(labels_text)
        for idx in range(len(vectorized_test_data)):
            vector_string = listToString(vectorized_test_data[idx])
            vector_string += f',{test_labels[idx]}\n'
            output_data.write(vector_string)

    # display_vectors(vectorized_training_data, vocab,
    #                 preprocessed_training_data)

    # Return the training labels and data for our bayesian network.
    return vectorized_training_data, vectorized_test_data, training_labels, test_labels, vocab


def plot_predictions(plot_data):
    # Get the x and y values for both our plots.
    x_axis = [25, 50, 75, 100]

    y_values_test = []
    for section in plot_data["test"].keys():
        y_values_test.append(plot_data["test"][section])

    y_values_train = []
    for section in plot_data["train"].keys():
        y_values_train.append(plot_data["train"][section])

    # Create a new figure and set its size
    plt.figure(figsize=(8, 4))

    # Plot the first line graph
    plt.subplot(2, 1, 1)  # (rows, columns, plot_number)
    plt.plot(x_axis, y_values_train, marker='o', color='blue')
    plt.title('Train Dataset Prediction Plot')
    plt.xlabel('Percentage Trained from Training Dataset')
    plt.ylabel('Prediction Accuracy Rate (%)')
    plt.xticks([25, 50, 75, 100], ["25%", "50%", "75%", "100%"])

    # Plot the second line graph
    plt.subplot(2, 1, 2)  # (rows, columns, plot_number)
    plt.plot(x_axis, y_values_test, marker='s', color='red')
    plt.title('Test Dataset Prediction Plot')
    plt.xlabel('Percentage Trained from Training Dataset')
    plt.ylabel('Prediction Accuracy Rate (%)')
    plt.xticks([25, 50, 75, 100], ["25%", "50%", "75%", "100%"])

    # Adjust the layout to prevent overlapping of titles and labels
    plt.tight_layout()

    # Display the line graphs
    plt.show()


def writeResults(data):

    with open("./results.txt", 'w') as output_data:
        # Print a header and a classification label for each loop.
        introduction = "Prediction Accuracy results for test and training datasets in 25% increments."
        output_data.write(introduction)

        training_introduction = "\n\nPrediction Accuracy of Training Documents"
        output_data.write(training_introduction)
        for idx in range(1, 5):
            prediction = f'\nTrained on {min(125*idx, 499)} documents: {data["train"][idx]}% accuracy'
            output_data.write(prediction)

        testing_introduction = "\n\nPrediction Accuracy of Testing Documents"
        output_data.write(testing_introduction)
        for idx in range(1, 5):
            prediction = f'\nTrained on {min(125*idx, 499)} documents: {data["test"][idx]}% accuracy'
            output_data.write(prediction)


def main():
    # Take in text files and outputs sentiment scores

    vectorized_training_data, vectorized_test_data, training_labels, test_labels, vocab = preprocessing_functionality()
    bayes_net = BayesClassifier()

    # We train over four increments and make predictions for the test and training datasets
    # on each prediction.
    end_increment = bayes_net.file_sections[0] + 1
    start_row = 0
    count = 0
    plot_data = {
        "train": {
            1: 0,
            2: 0,
            3: 0,
            4: 0
        },
        "test": {
            1: 0,
            2: 0,
            3: 0,
            4: 0
        }
    }

    while end_increment <= bayes_net.file_length + 1:
        bayes_net.train(vectorized_training_data, training_labels,
                        vocab, start_row, end_increment)
        count += 1

        # Then we classify vectorized test data:
        predictions_training_set = bayes_net.classify_text(
            vectorized_training_data, vocab)

        predictions_test_set = bayes_net.classify_text(
            vectorized_test_data, vocab)

        # Then we compare the predictions_training_set with
        # the actual results and print the comparison.
        print(f'\nAccuracy Results for iteration: {count}')
        training_prediction_accuracy = round(accuracy(
            predictions_training_set, training_labels))
        plot_data["train"][count] = training_prediction_accuracy
        print(
            f'Accuracy Rate for Training Set: {training_prediction_accuracy}%')

        test_prediction_accuracy = round(
            accuracy(predictions_test_set, test_labels))
        plot_data["test"][count] = test_prediction_accuracy
        print(
            f'Accuracy Rate for Testing Set: {test_prediction_accuracy}%\n')

        # Then we increment the iterations for the next predictions.
        start_row = end_increment
        end_increment += bayes_net.file_sections[0] + 1

    # With the predictions calculated, we plot them using matplotlib
    # and print them in the text file.
    plot_predictions(plot_data)
    writeResults(plot_data)

    return 1


if __name__ == "__main__":
    main()
