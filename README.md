# MammoDetect - Early Breast Cancer Detection

![Breast Cancer Prediction](images/breast-cancer-prediction.png)

Welcome to the **MammoDetect**, a cutting-edge web application that harnesses the power of deep learning to analyze mammogram images and predict the presence of breast cancer. With an intuitive user interface and advanced image processing techniques, this application aims to assist in the early detection of breast cancer and improve patient outcomes.

## Key Features

- **Upload Mammogram Images**: Easily upload your mammogram images through a user-friendly interface. The application supports PNG and JPEG image formats.

- **Real-Time Predictions**: The deep learning model integrated into the application provides real-time predictions on the uploaded mammogram images, ensuring quick and efficient results.

- **Visualize Cancerous Areas**: The application generates visualizations that highlight the cancerous areas within the mammogram images, assisting healthcare professionals in their analysis.

- **Risk Assessment**: Based on the prediction results, the application provides an assessment of the risk level associated with breast cancer, enabling informed decision-making and personalized patient care.

- **Interactive Results**: The application displays comprehensive prediction results, including the probability scores for different types of breast cancer, allowing for detailed analysis and interpretation.

- **Secure and Private**: The application prioritizes data privacy and security. Uploaded images are processed securely, and user data is handled with utmost confidentiality.

## How to Use

1. Access the application through your preferred web browser.

2. Click on the "Choose File" button to upload a mammogram image from your local device.

3. Once the image is uploaded, click on the "Predict" button to initiate the breast cancer prediction process.

4. The application will process the image using advanced deep learning techniques and display the prediction results in real-time.

5. Explore the visualization of cancerous areas within the image, along with a comprehensive assessment of the risk associated with breast cancer.

6. Use the provided information to guide further medical analysis and decision-making processes.

## Installation and Deployment

To deploy the MammoDetect locally, follow these steps:

1. Clone the repository:

```bash
git clone https://github.com/sanjyotpanure/MammoDetect
cd MammoDetect
```

2. Install the necessary dependencies using pip:

```bash
pip install -r requirements.txt
```

3. Download the pre-trained deep learning model file (model.h5) and save it in the **model** directory.

4. Launch the application by running the following command:

```bash
python app.py
```

5. Open your web browser and navigate to http://localhost:5000 to access the application.

## Screenshots

![Home Page](images/home.png)
Upload Mammogram Image

![Prediction Results](images/home.png)
Prediction Results and Visualization

## Technologies Used

1. Flask: A lightweight web framework used for handling HTTP requests and rendering dynamic HTML templates.

2. TensorFlow: The popular deep learning framework used for training and deploying the breast cancer prediction model.

3. PIL (Python Imaging Library): A library for image processing tasks, such as opening, manipulating, and saving images.

4. NumPy: A fundamental package for scientific computing with Python, utilized for array manipulation and mathematical operations.

5. scikit-image: A collection of algorithms for image processing and analysis, enabling advanced image segmentation and labeling.

6. matplotlib: A plotting library used to generate visualizations of the mammogram images and cancerous regions.

7. Werkzeug: A utility library that provides secure file uploading and other web-related functions.

## About
The **MammoDetect** is an innovative project that aims to leverage the power of artificial intelligence and image analysis techniques to assist healthcare professionals in the early detection and diagnosis of breast cancer. By providing accurate predictions and insightful visualizations, this application empowers medical experts and contributes to improved patient care.

## License
This project is licensed under the [MIT License](https://github.com/sanjyotpanure/MammoDetect/blob/master/LICENSE).