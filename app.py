from flask import Flask, request, render_template, url_for, jsonify
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

app = Flask(__name__)

def preprossing(image):
    '''image=Image.open(image)
    image = image.resize((150, 150))
    image_arr = np.array(image.convert('RGB'))
    image_arr.shape = (3, 224, 224, 3)
    '''
    image = Image.open(image)
    image = image.resize((224, 224))  # Resize to 224x224
    image_arr = np.array(image.convert('RGB'))
    image_arr = image_arr / 255.0  # Normalize pixel values to range [0, 1]
    image_arr = np.expand_dims(image_arr, axis=0)  # Add batch dimension
    return image_arr

classes = ['Bacterial_spot' ,'Early_blight', 'Late_blight' ,'Leaf_Mold' ,'Septoria_leaf_spot' ,'Spider_mites Two-spotted_spider_mite','Target_Spot','Tomato_Yellow_Leaf_Curl_Virus','Tomato_mosaic_virus','Healthy']
model=load_model("model_inception.h5")

@app.route('/')
def index():

    return render_template('index.html', appName="Leaf Care Tech")


@app.route('/predictApi', methods=["POST"])
def api():
    # Get the image from post request
    try:
        if 'fileup' not in request.files:
            return "Please try again. The Image doesn't exist"
        image = request.files.get('fileup')
        image_arr = preprossing(image)
        print("Model predicting ...")
        result = model.predict(image_arr)
        print("Model predicted")
        ind = np.argmax(result)
        prediction = classes[ind]
        print(prediction)
        return jsonify({'prediction': prediction})
    except:
        return jsonify({'Error': 'Error occur'})


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    print("run code")
    if request.method == 'POST':
        # Get the image from post request
        print("image loading....")
        image = request.files['fileup']
        print("image loaded....")
        image_arr= preprossing(image)
        print("predicting ...")
        result = model.predict(image_arr)
        print("predicted ...")
        ind = np.argmax(result)
        prediction = classes[ind]

        print(prediction)

        return render_template('index.html', prediction=prediction, image='static/IMG/plantIcon', appName="Leaf Care Tech")
    else:
        return render_template('index.html',appName="Leaf Care Tech")


if __name__ == '__main__':
    app.run(debug=True)