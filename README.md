# Traffic-Tweet-Classification-using-NLP

Traffic Detection NLP Project
This project aims to classify traffic-related tweets using various machine learning and deep learning models. The steps include data preprocessing, feature extraction, model training, and evaluation.

Libraries and Tools Used

Pandas: For data manipulation and analysis.
Scikit-learn: For machine learning model implementation.
NLTK: For natural language processing tasks.
Spacy: For advanced NLP tasks, including named entity recognition.
TensorFlow and Keras: For deep learning model implementation.
TensorFlow Hub: For pre-trained BERT models.

Data Preprocessing

Loading Data: The dataset is loaded from a CSV file.

Cleaning Data: A custom function is used to clean the text data. 
This involves:

Removing named entities.
Converting text to lowercase.
Removing punctuation and digits.
Removing stop words.
Lemmatizing the words.

Feature Extraction

TF-IDF Vectorization: The cleaned text data is converted into numerical features using TF-IDF vectorization.

Model Training and Evaluation

Splitting Data: The dataset is split into training and testing sets.

Training Models: Several models are trained, 
including:

Multinomial Naive Bayes
Logistic Regression
Support Vector Machine (SVM)
Random Forest Classifier
Convolutional Neural Network (CNN)
BERT (Bidirectional Encoder Representations from Transformers)

Evaluating Models: Each model is evaluated based on accuracy, loss, and classification reports.

Deep Learning Models

CNN: A Convolutional Neural Network is implemented using Keras, including embedding layers, convolutional layers, max-pooling layers, and LSTM layers.
BERT: A pre-trained BERT model is fine-tuned on the dataset using TensorFlow Hub. The model architecture includes dropout and dense layers for classification.

Results

Each model's performance is evaluated on the test dataset, and the accuracy and loss are reported.
Classification reports provide detailed performance metrics, including precision, recall, and F1-score.
This project demonstrates the application of various machine learning and deep learning techniques for text classification tasks in NLP.
