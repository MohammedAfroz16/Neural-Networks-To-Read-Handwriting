from flask import Flask, request, render_template
import cv2
import numpy as np
import os
import typing
from mltu.inferenceModel import OnnxInferenceModel
from mltu.utils.text_utils import ctc_decoder
from mltu.configs import BaseModelConfigs
from contours_sentences import segment_lines, segment_words  # Importing the functions

app = Flask(__name__)

class ImageToWordModel(OnnxInferenceModel):
    def __init__(self, char_list: typing.Union[str, list], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.char_list = char_list

    def predict(self, image: np.ndarray):
        # Resize and preprocess the image for prediction
        image = cv2.resize(image, self.input_shapes[0][1:3][::-1])
        image_pred = np.expand_dims(image, axis=0).astype(np.float32)
        preds = self.model.run(self.output_names, {self.input_names[0]: image_pred})[0]
        text = ctc_decoder(preds, self.char_list)[0]
        return text

def correct_sentence(sentence):
    # Implement your sentence correction logic
    return sentence  # Just returning the input for now

@app.route('/')
def index():
    # Clear previous results when going back to index
    return render_template('index(w_s).html', prediction=None, filename=None)

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return render_template('results.html', prediction='No file uploaded', filename=None), 400
    file = request.files['file']
    if file.filename == '':
        return render_template('results.html', prediction='No file selected', filename=None), 400

    # Process the image for word prediction
    image_path = os.path.join('static', 'uploads', file.filename)
    file.save(image_path)
    image = cv2.imread(image_path)

    # Load your word prediction model
    configs = BaseModelConfigs.load("C:/Users/mdafr/OneDrive/Desktop/flask_app/models/configs.yaml")
    model = ImageToWordModel(model_path=configs.model_path, char_list=configs.vocab)

    prediction_text = model.predict(image)
    return render_template('results.html', filename=file.filename, prediction=prediction_text, image_path=image_path)

@app.route('/upload_sentence', methods=['POST'])
def upload_sentence():
    if 'file' not in request.files:
        return render_template('results.html', prediction='No file uploaded', filename=None), 400
    file = request.files['file']
    if file.filename == '':
        return render_template('results.html', prediction='No file selected', filename=None), 400

    # Process the image for sentence prediction
    image_path = os.path.join('static', 'uploads', file.filename)
    file.save(image_path)
    image = cv2.imread(image_path)

    # Load your sentence prediction model
    configs = BaseModelConfigs.load("C:/Users/mdafr/OneDrive/Desktop/flask_app/models/configs.yaml")
    model = ImageToWordModel(model_path=configs.model_path, char_list=configs.vocab)

    line_images = segment_lines(image)

    # Ensure line_images is not None and is a list
    if line_images is None or not line_images:
        return render_template('results.html', prediction='No lines segmented from the image', filename=file.filename), 400

    all_predicted_words = []

    for line_img in line_images:
        word_images = segment_words(line_img)

        # Ensure word_images is not None and is a list
        if word_images is None or not word_images:
            all_predicted_words.append("")  # Add an empty string for this line if no words found
            continue  # Skip to the next line if no words were segmented

        predicted_words = []

        for word_img in word_images:
            word_prediction = model.predict(word_img)
            predicted_words.append(word_prediction)

        all_predicted_words.append(" ".join(predicted_words))

    final_sentence = " ".join(all_predicted_words)
    corrected_sentence = correct_sentence(final_sentence)

    return render_template('results.html', filename=file.filename, prediction=corrected_sentence, image_path=image_path)

if __name__ == "__main__":
    app.run(debug=True)
