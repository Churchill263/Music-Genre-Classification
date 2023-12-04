# Music-Genre-Classification

Music Genre Classification

This project is designed to perform music genre classification using a deep learning model built with TensorFlow and Keras. The classification model is trained on audio features extracted from a curated dataset of music samples.

Project Overview

Music genre classification is a common problem in audio signal processing and machine learning. In this project, we leverage the power of deep learning to predict the genre of a given music sample. The dataset used for training and testing contains audio features such as Mel-frequency cepstral coefficients (MFCCs), which are extracted using the librosa library.

Project Structure

music_genre_classification.ipynb: Jupyter Notebook containing the code for data preprocessing, model training, and evaluation.
requirements.txt: List of project dependencies.
encoder.pkl: Pickle file containing the label encoder used for mapping genre labels to numerical values.
saved_models/: Directory containing saved models during training.
audio_set_path: Path to the directory containing the original audio files (replace with your own path).
metadata.csv: CSV file containing metadata and features for the audio dataset.

Setup
1. Clone the Repository
(Copy code
git clone https://github.com/your-username/music-genre-classification.git
cd music-genre-classification)

2. Create a Virtual Environment
(python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate)

3. Install Dependencies
(pip install -r requirements.txt)

4. Run the Google Colab File
(Google Colab music_genre_classification.ipynb)

Training the Model
Follow the steps outlined in the Google Colab (music_genre_classification.ipynb) to load the dataset, preprocess the audio features, train the deep learning model, and evaluate its performance.

Model Files
The trained model is saved in the saved_models/ directory with a filename that includes the timestamp of when it was saved. This directory is useful for versioning and tracking model improvements.

Label Encoder
The label encoder used to map genre labels to numerical values is saved in the encoder.pkl file. This file is necessary for decoding predicted genre labels during inference. Ensure that this file is accessible when deploying the model for predictions.

Acknowledgements
The dataset used for this project is available at [link to dataset].
Special thanks to the authors of the libraries used in this project: NumPy, Pandas, librosa, TensorFlow, and scikit-learn.

Additional Notes
Feel free to customize the project structure and adapt the code to your specific needs. If deploying the model as a web application, consider using frameworks like Streamlit or Flask. Ensure that you have the necessary permissions for the audio dataset and comply with any licensing restrictions associated with the data.

Link to the Youtube Video:  https://youtu.be/2DAsiRWRKGQ



