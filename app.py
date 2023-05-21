from flask import Flask, render_template, request
# from flask_ngrok import run_with_ngrok
from PIL import Image
from io import BytesIO
import numpy as np
import tensorflow as tf
from skimage import measure
from scipy import ndimage
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from werkzeug.utils import secure_filename
import uuid
import os
import traceback

app = Flask(__name__)
# run_with_ngrok(app)
model = tf.keras.models.load_model('model/nmodel.h5')
class_name = ['Benign with Density: 1', 'Malignant with Density: 1', 'Benign with Density: 2', 'Malignant with Density: 2',
              'Benign with Density: 3', 'Malignant with Density: 3', 'Benign with Density: 4', 'Malignant with Density: 4']

# Define a function to generate the mask for the cancerous area
def generate_mask(image):
    # Convert the image to grayscale
    gray = np.mean(image, axis=2)
    # Threshold the image to obtain the cancerous area
    mask = gray > np.max(gray) * 0.5
    # Fill any holes in the mask
    mask = ndimage.binary_fill_holes(mask).astype(int)
    # Remove small objects from the mask
    mask = measure.label(mask)
    props = measure.regionprops(mask)
    areas = np.array([p.area for p in props])
    mask = np.isin(mask, np.where(areas >= np.max(areas) * 0.5)[0] + 1)
    # Reshape the mask to a 3D array with a single channel
    mask = np.expand_dims(mask, axis=-1)
    return mask

# Define a function to visualize the cancerous area
def visualize_cancer(image, mask):
    # Apply the mask to the image to obtain the cancerous region
    cancer_region = np.multiply(image, mask)
    # Generate a heatmap of the cancerous region
    heatmap = np.mean(cancer_region, axis=-1)
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(image)
    ax.imshow(heatmap, alpha=0.5, cmap='cool', interpolation='nearest')
    # Instead of saving the figure, we can return it as a PNG image
    buf = BytesIO()
    # plt.axis('off')
    plt.savefig(buf, format='png')
    # plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    plt.close()
    buf.seek(0)
    return buf.getvalue()

def predict_cancer(image):
    img = image.resize((224, 224))  # resize the image
    img_array = np.array(img) / 255.0  # normalize the image
    img_array = tf.expand_dims(img_array, 0)
    predictions = model.predict(img_array)
    return {class_name[i]: float(predictions[0][i]) for i in range(8)}


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/info')  
def info ():  
    return render_template("info.html") 

@app.route('/predict', methods=['POST'])
def predict():
    try:
        file = request.files['file']
        
        # Check if a file is uploaded
        if not file:
            return render_template('index.html', message='No file uploaded.')
        
        unique_filename = str(uuid.uuid4())
        filename = secure_filename(file.filename)
        _, extension = os.path.splitext(filename)
        img_filename = unique_filename + extension
        img_path = os.path.join('static/temp/', img_filename)
        file.save(img_path)

        img = Image.open(img_path)
        
        # Check if the uploaded file is a valid mammogram image
        if img.format != 'PNG' and img.format != 'JPEG':
            os.remove(img_path)  # Remove the uploaded file
            return render_template('index.html', message='Invalid file format. Please upload a PNG or JPEG image.')
        predictions = predict_cancer(img)
        pred_idx = np.argmax(list(predictions.values()))

        mask = generate_mask(np.array(img))
        mask_3d = np.repeat(mask, 3, axis=-1)
        if pred_idx in [1, 3, 5, 7]:
            if predictions[class_name[pred_idx]] >= 0.5:
                vis_image = visualize_cancer(np.array(img), mask_3d)
                vis_filename = unique_filename + '_vis.png'
                vis_file_path = os.path.join('static/temp/', vis_filename)
                with open(vis_file_path, 'wb') as f:
                    f.write(vis_image)
                return render_template('result.html', prediction=predictions, img_filename=img_filename,
                                       vis_filename=vis_filename,
                                       message="The results show the presence of malignancy in the breast tissue, suggesting that the patient may have a risk of breast cancer.")
            else:
                return render_template('result.html', prediction=predictions, img_filename=img_filename,
                                       message="The results show the absence of malignancy in the breast tissue, suggesting that the patient has a low risk of breast cancer.")
        else:
            return render_template('result.html', prediction=predictions, img_filename=img_filename,
                                   message="The image is classified as benign. Hence, there is no risk of breast cancer.")
    except FileNotFoundError:
        return render_template('index.html', message='Model not found.')
    except Exception as e:
        traceback.print_exc()  # Print the traceback of the exception
        return render_template('index.html', message='Uploaded image is not in correct format! Please upload valid image.')


if __name__ == '__main__':
    app.run(debug=False)