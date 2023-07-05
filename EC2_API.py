from io import BytesIO
import numpy as np
import tensorflow as tf
from PIL import Image
from flask import Flask, jsonify, request
#from keras.preprocessing.utils import img_to_array
from tensorflow_addons.metrics import F1Score
# Load the model
model = tf.keras.models.load_model('best_model.h5', custom_objects={'FixedDropout': tf.keras.layers.Dropout, 'Addons>F1Score': F1Score})

class_names = ['healthy apple',
 'healthy bell pepper',
 'healthy corn (maize)',
 'healthy grape',
 'healthy potato',
 'unhealthy apple',
 'unhealthy bell pepper',
 'unhealthy corn (maize)',
 'unhealthy grape',
 'unhealthy potato']

def prepare_image(img):
    try:
        img = img.resize((224,244))
        img = np.array(img)
        img = img / 255.0
        img = np.expand_dims(img, axis=0)

    # Predict the class and confidence score
        answer = model.predict(img)
        predicted_class = class_names[np.argmax(answer[0])]
        print(f'The predicted class is:  {predicted_class}')
        confidence = float(np.max(answer[0]))
        return {
            'class': predicted_class,
            'confidence': confidence
        }
    except Exception as e:
        return {'error': str(e)}

app = Flask(__name__)

@app.route('/ping', methods=['GET'])
def ping():
    return "Hello, I am alive"

@app.route('/predict', methods=['POST'])
def infer_image():
    if 'file' not in request.files:
        return jsonify(error="Please try again. The Image doesn't exist")

    file = request.files.get('file')
    img_bytes = file.read()
    img = Image.open(BytesIO(img_bytes))
    result = prepare_image(img)
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0')