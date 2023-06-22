import argparse

import cv2
import imutils
import numpy as np
from imutils.contours import sort_contours
import matplotlib.pyplot as plt
from matplotlib import use as mpl_use
import tensorflow as tf

mpl_use('TkAgg')

if __name__ == '__main__':
    # Parse the command line arguments.
    parser = argparse.ArgumentParser(description='Recognize a handwritten text from an input image.')
    parser.add_argument(dest='file_path', metavar='file-path', type=str, help='The file path to the input image.')
    args = parser.parse_args()

    # Load the image
    image = cv2.imread(args.file_path)

    # Convert the image from BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.imshow(image), plt.title('Original image'), plt.show()

    # Convert the image to grayscale
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Blur the grayscale image using Gaussian blur
    blured_image = cv2.GaussianBlur(grayscale_image, (5, 5), 0)

    # Threshold the grayscale image using Otsu's method
    _, thresh = cv2.threshold(grayscale_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    plt.imshow(thresh, cmap='gray'), plt.title('Thresholded image'), plt.show()

    # Dilate the thresholded image
    kernel_size = (10, 5)
    morphological_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    iterations = 5
    dilated_image = cv2.morphologyEx(thresh, cv2.MORPH_DILATE, morphological_kernel, None, None, iterations,
                                     cv2.BORDER_REFLECT101)
    plt.imshow(dilated_image, cmap='gray'), plt.title('Dilated image'), plt.show()

    # Contour the dilated image
    contours, _ = cv2.findContours(dilated_image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Sort the contours according to the lines of text appearing in the thresholded image
    sum_of_rows = np.sum(dilated_image, axis=1)
    start_index = 0
    lines = []
    flag = True
    threshold = 0
    for index, value in enumerate(sum_of_rows):
        change = (value > threshold)
        if change == flag:
            if value <= threshold:
                lines.append((start_index, index))
                start_index = index + 1
            flag = not flag

    # Detect the line contours
    line_contours = []
    for index, contour in enumerate(contours):
        x, y, w, h = cv2.boundingRect(contour)
        for i, line in enumerate(lines):
            if line[0] <= y <= line[1]:
                line_contours.append([line[0], x, index])
                break

    sorted_contours = [contours[index] for line, x, index in sorted(line_contours)]
    text = []
    for i, contour in enumerate(sorted_contours):
        # Isolate the word ROI
        x, y, w, h = cv2.boundingRect(contour)
        word_roi = grayscale_image[y:y + h, x:x + w]

        # Blur the ROI using Gaussian blur
        word_image = cv2.GaussianBlur(word_roi, (1, 1), 0)

        # Threshold the word image using Otsu's method
        _, word_thresh = cv2.threshold(word_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Contour the thresholded word image
        character_contours = cv2.findContours(word_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Grab and sort the contours
        character_contours = imutils.grab_contours(character_contours)
        sorted_character_contours = sort_contours(character_contours, method='left-to-right')[0]

        word = []
        for j, character_contour in enumerate(sorted_character_contours):
            # Isolate the character ROI
            x1, y1, w1, h1 = cv2.boundingRect(character_contour)
            character_roi = word_thresh[y1:y1 + h1, x1:x1 + w1]

            word.append((character_roi, x + x1, y + y1, w1, h1))

        text.append(word)

    characters = []
    schema = []
    for word in text:
        for character in word:
            thresh, x, y, w, h = character

            # Resize the largest dimension of the input size
            (tH, tW) = thresh.shape
            if tW > tH:
                thresh = imutils.resize(thresh, width=28)
            else:
                thresh = imutils.resize(thresh, height=28)

            # Find how much is needed to pad
            (tH, tW) = thresh.shape
            dX = int(max(0, 28 - tW) / 2.0)
            dY = int(max(0, 28 - tH) / 2.0)

            # Pad the image and force 28 x 28 dimensions
            padded = cv2.copyMakeBorder(thresh, top=dY, bottom=dY, left=dX, right=dX, borderType=cv2.BORDER_CONSTANT,
                                        value=(0, 0, 0))
            padded = cv2.resize(padded, (28, 28))

            # Reshape and rescale the padded image for the model
            padded = padded.astype('float32') / 255.0
            padded = np.expand_dims(padded, axis=-1)

            # Append the image and bounding box data in the list of characters
            characters.append((padded, (x, y, w, h)))
            # Append the marker 'X' in the schema
            schema.append('X')

        # Append the market ' ' (blank space) in the schema
        schema.append(' ')

    # Plot the isolated characters
    n_cols = 10
    n_rows = int(np.floor(len(characters) / n_cols) + 1)
    fig = plt.figure(figsize=(1.5 * n_cols, 1.5 * n_rows))
    for i, char in enumerate(characters):
        ax = plt.subplot(n_rows, n_cols, i + 1)
        ax.imshow(char[0][:, :, 0], cmap='gray', aspect='auto')
    plt.tight_layout()
    plt.show()

    boxes = [box[1] for box in characters]
    chars = np.array([character[0] for character in characters], dtype='float32')

    # Load the model
    model_path = 'optical-character-recognizer'
    print("Loading deep neural network model...")
    model = tf.keras.models.load_model(model_path)

    # Classify the characters using the handwriting recognition model
    predictions = model.predict(chars)

    # Define the list of label names
    label_names = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'

    for (prediction, (x, y, w, h)) in zip(predictions, boxes):
        # Find the index of the label with the largest corresponding probability, then extract its probability and label
        index = int(np.argmax(prediction))
        probability = prediction[index]
        label = label_names[index]

        # Draw the prediction on the image and it's probability
        label_text = f'{label},{probability * 100:.1f}%'
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(image, label_text, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # Show the image
    plt.figure(figsize=(15, 10))
    plt.imshow(image)
    plt.show()

    # Convert the text from the image into string
    string = ''
    for prediction in predictions:
        if schema.pop(0) == ' ':
            string = string + ' '
            schema.pop(0)
        index = int(np.argmax(prediction))
        label = label_names[index]
        string = string + label

    print(string)
