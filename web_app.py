from flask import Flask, render_template, request
from keras.models import load_model
from keras.preprocessing import image
from keras.saving.model_config import model_from_json
import tensorflow as tf
import numpy as np

print(tf.__version__)
app = Flask(__name__)

dic = {0: 'Glaucoma, consult a doctor', 1: 'No Glaucoma'}

# load json and create model
json_file = open('ImageClassifier.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)

# load weights into new model
model.load_weights('ImageClassifier.h5')
print("Loaded model from disk")
# model = load_model('ImageClassifier.h5')

model.make_predict_function()


def predict_label(img_path):
    i = tf.keras.utils.load_img(img_path, target_size=(150, 150))
    i = tf.keras.utils.img_to_array(i) / 255.0
    i = i.reshape(1, 150, 150, 3)
    predict_x = model.predict(i)
    p = np.argmax(predict_x, axis=1)
    return dic[p[0]]


# routes
@app.route("/", methods=['GET', 'POST'])
def main():
    return render_template("index.html")


@app.route("/submit", methods=['GET', 'POST'])
def get_output():
    if request.method == 'POST':
        img = request.files['my_image']
        img_path = "static/" + img.filename
        img.save(img_path)
        p = predict_label(img_path)
    return render_template("index.html", prediction=p, img_path=img_path)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
    # app.run(debug=True)
