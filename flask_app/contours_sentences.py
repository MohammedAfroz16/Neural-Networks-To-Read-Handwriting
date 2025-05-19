import cv2
import numpy as np
import os
import typing
from mltu.inferenceModel import OnnxInferenceModel
from mltu.utils.text_utils import ctc_decoder
import language_tool_python
import spacy

class ImageToWordModel(OnnxInferenceModel):
    def __init__(self, char_list: typing.Union[str, list], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.char_list = char_list

    def predict(self, image: np.ndarray):
        image = cv2.resize(image, self.input_shapes[0][1:3][::-1])
        image_pred = np.expand_dims(image, axis=0).astype(np.float32)
        preds = self.model.run(self.output_names, {self.input_names[0]: image_pred})[0]
        text = ctc_decoder(preds, self.char_list)[0]
        return text

def segment_lines(sentence_image):
    gray = cv2.cvtColor(sentence_image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (sentence_image.shape[1] // 2, 1))
    dilated = cv2.dilate(binary, kernel, iterations=5)

    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    line_images = []

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if h > 20:
            line_img = sentence_image[y:y + h, x:x + w]
            line_images.append((line_img, y))

    line_images = sorted(line_images, key=lambda x: x[1])
    return [img for img, _ in line_images]

def segment_words(line_image):
    gray = cv2.cvtColor(line_image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (8, 8))
    dilated = cv2.dilate(binary, kernel, iterations=1)
    eroded = cv2.erode(dilated, kernel, iterations=1)

    contours, _ = cv2.findContours(eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    word_images = []
    bounding_boxes = []

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w > 10 and h > 10:
            word_img = line_image[y:y+h, x:x+w]
            word_images.append(word_img)
            bounding_boxes.append((x, y, w, h))

    sorted_boxes = sorted(bounding_boxes, key=lambda box: box[0])
    sorted_word_images = [word_images[bounding_boxes.index(box)] for box in sorted_boxes]

    return sorted_word_images

def correct_sentence(sentence):
    tool = language_tool_python.LanguageTool('en-US')
    matches = tool.check(sentence)
    nlp = spacy.load('en_core_web_sm')
    doc = nlp(sentence)

    corrected_words = []
    for token in doc:
        if token.pos_ == "NOUN":
            corrected_words.append(token.text)
        else:
            corrected_word = language_tool_python.utils.correct(token.text, [m for m in matches if m.offset <= len(token.text) and m.offset + len(token.text) > 0])
            corrected_words.append(corrected_word)

    corrected_sentence = ' '.join(corrected_words)
    return corrected_sentence

def predict_sentence(model, sentence_image):
    line_images = segment_lines(sentence_image)
    all_predicted_words = []

    output_dir = "segmented_words"
    os.makedirs(output_dir, exist_ok=True)

    for i, line_img in enumerate(line_images):
        line_output_path = os.path.join(output_dir, f"line_{i + 1}.png")
        cv2.imwrite(line_output_path, line_img)
        print(f"Saved line image: {line_output_path}")

        word_images = segment_words(line_img)
        predicted_words = []

        for j, word_img in enumerate(word_images):
            word_output_path = os.path.join(output_dir, f"line_{i + 1}_word_{j + 1}.png")
            cv2.imwrite(word_output_path, word_img)

            word_prediction = model.predict(word_img)
            print(f"Predicted Word: {word_prediction}")
            predicted_words.append(word_prediction)

        all_predicted_words.append(" ".join(predicted_words))

    final_sentence = "\n".join(all_predicted_words)
    return final_sentence

if __name__ == "__main__":
    from mltu.configs import BaseModelConfigs

    configs = BaseModelConfigs.load("models/configs.yaml")
    model = ImageToWordModel(model_path=configs.model_path, char_list=configs.vocab)

    sentence_image_path = "C:/Users/mdafr/OneDrive/Pictures/Screenshots/sevenlines.png"
    sentence_image = cv2.imread(sentence_image_path)

    if sentence_image is not None:
        final_sentence = predict_sentence(model, sentence_image)
        print("Final Predicted Sentence:\n", final_sentence)
    else:
        print(f"Error: The image file {sentence_image_path} does not exist.")
