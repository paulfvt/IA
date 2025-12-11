# IA
# Find Your MBTI: Not by How You Think, but by Who You Are


### Members: 

| Name | Organization | Email | 
|--------------------------|------------------------|----------------------------------------|
| Paul Fievet | Computer of Science | paul.fievet@edu.ece.fr | 
| Thomas Pernod | Computer of Science | thomas.pernod@edu.ece.fr | 
| Mélissande LEMAIRE-VALLE | Computer of Science | melissande.lemairevalle@edu.ece.fr | 
| Manon QUESNE | Computer of Science | manon.quesne@edu.ece.fr | 
| Bertille METIER | Computer of Science | bertille.metier@edu.ece.fr|


### 1. Introduction: 

We decided to create an AI capable of analyzing a person’s true personality not just how they perceive themselves, but who they really are deep down. Our goal is to go beyond the traditional MBTI tests, which often rely on self-reported answers and can be biased by mood, social desirability, or cultural influences. We want our AI to interpret behavioral cues, word choice, and emotional patterns to build a more authentic psychological profile.
This idea also comes from an observation in South Korea, where MBTI has become extremely popular almost a social trend. People use it in dating, work, and even daily conversations, but often in a superficial way. We wanted to take that concept further: rather than just saying “I’m an INFP” or “I’m an ESTJ,” our project aims to reveal what truly defines someone beyond the label. 
At the end, we want the user to be able to input information about their thoughts, habits, or reactions and get back not only a personality type but also a deeper interpretation of their traits. The AI could highlight strengths, weaknesses, or even psychological tendencies that might need attention (though we’re not positioning ourselves as a medical or diagnostic tool). 
Our ambition is to build something that helps people understand themselves better, in a genuine, data-driven, and introspective way not just follow a social trend but uncover their real inner identity.   

### 2. Dataset:  

To train our hybrid personality analysis model, we rely on two Kaggle datasets and one internal dataset that provides user interface questions.
Each dataset plays a specific role in capturing different aspects of personality: writing style, behavioral choices, and real user input.  

##### A) Text Dataset – MBTI_500 (Kaggle) 

This dataset provides the foundation for our NLP-based personality model. 

 It contains: 

- 500 users,
- a large block of text per user (approximately 20 forum posts merged),
- a corresponding MBTI type. 

The text samples contain strong indicators of personality, such as: 

- tone and emotional expression,
- vocabulary richness,
- abstract vs. concrete language,
- assertiveness,
- use of personal pronouns,
- interaction style (supportive, analytical, argumentative…). 
 
These linguistic patterns correlate with MBTI traits (I/E, N/S, T/F, J/P). 

How do we use it? 

- We extract the post column as the input text. 
- We encode each text using SentenceTransformer – all-MiniLM-L6-v2, producing 384-dimension vectors.
- These embeddings are used to train an SVM classifier with an RBF kernel, which predicts the MBTI from writing style.

This dataset enables the system to generate a personality profile even without QCM answers, purely based on how a user writes.
 

##### B) QCM Dataset – 60k MBTI Responses (Kaggle) 

This dataset provides the structure for our behavior-based model. 

 It includes: 

- over 60,000 questionnaire responses,
- dozens of personality statements rated from 1 to 5,
- a “personality” column indicating the MBTI result. 

Why is this dataset useful?
It is significantly larger and more structured than the text dataset.
It captures self-reported behaviors such as: 

- social comfort (E/I),
- decision-making logic (T/F),
- organization habits (J/P),
- focus on details vs. big picture (S/N). 

How do we use it? 

- We clean the dataset and export it as 16P_converted.csv.
- We remove the “Response Id” column and keep only the numerical question responses.
- We scale all features using StandardScaler.
- We train a GradientBoostingClassifier (300 trees, learning_rate=0.05). 

Because of the large dataset size, the model is more stable and generalizes well. 

 

#### C) Internal Dataset – questions.json 

This file contains all the questions presented to the user in our Streamlit interface. 

It includes two types of inputs: 

- QCM questions (Likert scale sliders 1–5).
- Open-ended questions (text fields).

Role in the system? 

- The QCM questions mirror the structure of the training dataset, allowing the gradient boosting model to make accurate predictions.
- The open-ended questions give personal text samples that are processed by the NLP model.
- Both types of answers are combined later through a model fusion system.
- This dataset acts as the bridge between the training data and real user interaction. 

#### D) Complementarity of the Datasets 

The strength of our system lies in using two different psychological signals: 

| Dataset         | What it captures   | Model                                     |
|-----------------|--------------------|-------------------------------------------|
| MBTI_500        | Natural language, emotions, communication style  | SentenceTransformer & Logistic regression |
| 16P_converted   | Structured behaviors, preferences, habits  | RandomForest                              |
| questions       | Real user input for prediction  |                                           | 

 

Why does this combination matter? 

A single dataset cannot fully describe someone’s personality. 

By combining behavioral answers and writing patterns, our AI provides a more complete analysis than traditional MBTI tests. 

- The text model reveals how the person naturally expresses themselves.
- The QCM model reveals how the person perceives their own behavior.
- The fusion model blends both perspectives to produce a balanced MBTI prediction. 



### 3. Methodology :  

The methodology is based on two complementary supervised models, trained separately and then fused at the probability level to predict one of the sixteen MBTI types.  

For the QCM part, each user is described by 60 answers on a 1-to-5 scale (one feature per question). These are tabular data, of moderate size, with potentially non-linear relationships between questions. We therefore chose a RandomForestClassifier from scikit-learn rather than a linear model. The random forest aggregates many decision trees trained on sub-samples of the data and sub-sets of features, which allows it to capture complex interactions between questions (for example, combinations that are typical of a given MBTI type) while remaining robust to noise and redundant variables. Since trees are not sensitive to feature scaling, we do not apply any normalization to these 60 features, which simplifies the pipeline. The train_models.py script reads 16P_converted.csv, validates the columns, splits data into train/test sets, trains the forest, and saves the model with joblib.  

For the text part, we handle open-ended answers in two steps: semantic encoding and then classification. All of a user’s text answers are concatenated into a single document, which is then encoded by a pre-trained SentenceTransformer model (all-MiniLM-L6-v2). This transformer network produces a dense vector that summarizes the style, content, and preferences expressed in the text. On top of these embeddings, we train a multinomial logistic regression (LogisticRegression from scikit-learn): it is a linear model, simple to explain, but sufficient because the non-linear part is already handled by the transformer. As for the QCM, train_models.py reads MBTI_500.csv, encodes the texts, performs a train/test split, trains the logistic regression, and then saves the text model.  

The Streamlit application (ui.py) performs inference by combining both models. It reads questions.json to dynamically generate the 60 QCM sliders and 8 open questions. The QCM answers are reconstructed into a pandas DataFrame with the same column names as those used during training, then passed to the RandomForest model, which returns a probability distribution over the 16 types. The text answers are concatenated, encoded with the same SentenceTransformer, and then fed into the text model, which returns a second probability distribution. We then apply a “late fusion” approach:  

fused_probas = weight_qcm * prob_qcm + (1 - weight_qcm) * prob_text

where weight_qcm is chosen by the user in the sidebar. After renormalization, we take the MBTI type with the highest probability and also display the top 5 types, which provides a more nuanced view of the result.  

(Photo on PDF)

### 4. Graph Analysis

##### General project architecture: 

📁 projet_MBTI/
│

├── 📁 data/

│   ├── 16P_converted.csv

│   └── MBTI_500.csv

│   └── questions.json

│

├── 📁 models/

│   ├── qcm_model_rf.joblib

│   └── qcm_columns.joblib

│   └── text_model.joblib

│

├── 📁 src/

│   ├── train_models.py

│   └── ui.py

│

└── requirements.py  

##### Diagram of how the MBTI AI project works: 






### 5. Related Work 

In the context of this MBTI prediction system project, our methodology incorporated a dual strategy, combining fundamental academic resources with generative Artificial Intelligence tools. 

##### A) Optimization via Generative AI

To ensure the efficiency, and optimization of our code, we used several Artificial Intelligence as tools for continuous verification and improvement: 

- Gemini
- ChatGPT
- Claude

These AI platforms were utilized not to generate the entire project, but to serve as critical coding partners. Their primary role was to perform comparative analyses on different implementations of functionalities. This allowed us to study multiple coding alternatives for each module to determine the most performant, clear, and maintainable solution.

This multi-AI approach enables us to constantly evaluate whether the chosen coding path was the "best possible path" or if there were substantial improvements to integrate. 

##### B) Academic Resources and Data

The foundation of our work was built upon rigorous teaching and reliable data sources. 

We used the Kaggle platform to acquire the datasets necessary for training our models, including MBTI profile data and questionnaire response data. 

We also used the courses and documents available on LMS by the professor for the structural and theoretical design of the project, including the understanding of inheritance principles, polymorphism, and Machine Learning concepts. 

##### C) System Architecture and Key Libraries

Our MBTI prediction project was developed using several Python libraries. It includes essential tools for data management (pandas, numpy), Machine Learning libraries (scikit-learn, sentence-transformers, joblib), an interactive user interface via Streamlit, and a high-performance API deployment architecture using FastAPI and Uvicorn. 


| Category                     | Tool                       | Usage in the Code                                                                                                                                                                             | Role                                                                                      |
|------------------------------|----------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------|
| Data & Computation           | pandas                     | Reading source files (pd.read_csv), data preparation, and crucially, creating the QCM response DataFrame from user input.                                                                     | Fundamental library for data manipulation and analysis using DataFrames.                  |
|                              | numpy                      | Manipulating embedding vectors, managing probability arrays, and performing calculations during fusion (np.argmax(), weighted operations).                                                    | The foundation for scientific computing in Python, providing high-performance arrays.     |
|                              | json                       | Loading the questionnaire content from the questions.json file.                                                                                                                               | Allows reading and writing data in the JSON format.                                       |
| Machine Learning Core        | scikit-learn (via modules) | Provides algorithms (GradientBoostingClassifier, SVC), preprocessing (StandardScaler, Pipeline), and evaluation tools.                                                                        | The go-to library for classical Machine Learning (models, preprocessing, metrics).        |
|                              | sentence-transformers      | Loading a model (e.g., all-MiniLM-L6-v2) to transform the user's free text into numerical vectors.                                                                                            | Specialized library for creating sentence embeddings (vector representations of meaning). |
|                              | joblib                     | Saving (dump) and loading (load) the trained models (.joblib) and the preprocessor.                                                                                                           | Tool for fast serialization of large Python objects, ideal for ML models.                 |
| API Development (Backend)    | fastapi                    | Used to define the endpoints that will receive user data and return the MBTI predictions.                                                                                                     | Modern, high-performance Python framework for building web APIs.                          |
|                              | uvicorn[standard]          | Used to run and serve the FastAPI application, making it accessible via HTTP.                                                                                                                 | Ultra-fast ASGI Web Server (Asynchronous Server Gateway Interface).                       |
|                              | tensorflow / tf-keras      | Used if you had chosen deep neural network models. Often included as an indirect dependency for advanced text embedding models.                                                               | Frameworks for Deep Learning.                                                             |
| User Interface & Resources   | Streamlit                  | Creating the user interface (ui.py): sliders, text areas, weight management, and displaying results.                                                                                          | Provides tools for interacting with the operating system.                                 |
|                              | os (Operating System)                        | Managing file paths (os.path.join()) and creating necessary directories (os.makedirs()).                                                                                                      | Allows reading and writing data in the JSON format.                                       |



##### DATASET PATHS: 

https://www.kaggle.com/datasets/zeyadkhalid/mbti-personality-types-500-dataset 

https://www.kaggle.com/datasets/anshulmehtakaggl/60k-responses-of-16-personalities-test-mbt 

 