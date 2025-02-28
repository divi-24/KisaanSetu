<a id="top"></a>
<div style="display:none;" align="center">
<h1><font size="10"> KisaanSetu </font></h1>

<!-- repo intro -->

</div>
<div align="center">

<h3><font size="4">KisaanSetu platform is a comprehensive web-based tool where users can access various machine learning models for making accurate predictions related to agriculture. It offers solutions for crop management, soil health assessment, pest control, and more.</h3>
<br>
Make sure you star the repository and show your love to us💗
</font>
<br>
<br>
<p>

## Project Description

KisaanSetu platform is a comprehensive web-based tool where users can access various machine learning models for making accurate predictions related to agriculture. It offers solutions for crop management, soil health assessment, pest control, and more.

It implements machine learning algorithms to implement 3 basic functionalities:
### 1. Fertilizer Prediction
Aims to predict the appropriate fertilizer based on environmental and soil conditions. The dataset contains various features like temperature, humidity, moisture, soil type, crop type, and the levels of nitrogen, potassium, and phosphorus in the soil. The model aims to recommend the correct fertilizer to use, improving crop yield and soil health.
 
### Dataset: 
Fertilizer Prediction.csv (Uploaded under notebooks)
 
### Model Development :
A Random Forest Classifier was chosen as the primary model due to its robustness and high accuracy in classification tasks. The dataset was split into training and testing sets in an 80:20 ratio.
Key steps:

#### Label Encoding: 
Categorical variables (Soil Type, Crop Type, and Fertilizer Name) were encoded using LabelEncoder.
#### Model Training: 
A Random Forest model was trained using the training data.
#### Hyperparameter Tuning: 
A grid search with cross-validation was applied to find the optimal parameters for the Random Forest model.

### 2. Crop Prediction
Develop a machine learning-based crop recommendation system that uses various classification algorithms to predict the optimal crop for farming based on soil and environmental factors. The model takes inputs such as Nitrogen (N), Phosphorus (P), Potassium (K), temperature, humidity, pH level, and rainfall, and outputs the most suitable crop for specific conditions.
#### Dataset:
Crop_recommendation.csv

#### Model Training and Results
Four different models were trained on the dataset to predict the crop:
The results of each model are as follows:

1. Logistic Regression: 96.18%
2. Decision Tree Classifier: 97.82%
3. Gaussian Naive Bayes: 99.45%
4. Random Forest Classifier: 99.45%
The final model selected for deployment is the Random Forest Classifier.

### 3. Soil Quality Prediction 
Implements machine learning models to classify soil quality based on various features like nitrogen content, pH levels, organic carbon, and other nutrients. The goal of the model is to predict the quality of soil using logistic regression and a Support Vector Machine (SVM) model.

#### Dataset:
Soil_Quality.csv (Uploaded under notebooks)

#### Model Traning and Results
1. Logistic Regression
Logistic Regression is used to model the soil quality based on the provided features. The dataset is split into training and testing sets, and the logistic regression model is trained on the training data.

2. Support Vector Machine (SVM)
A Support Vector Machine with an RBF (Radial Basis Function) kernel is trained as an alternative model. The SVM model aims to find the decision boundary that best separates different soil quality classes.

3. Performance Evaluation
The performance of both models is evaluated using accuracy. The accuracy of each model is calculated by comparing the predicted soil quality labels with the actual labels in the test dataset.

### 4. Yield Prediction
Aims to develop a machine learning-based model for predicting crop yields based on various environmental and agricultural factors. The primary objective of this project is to create a model that predicts the total crop yield for a given region using data such as Area and type of crop, Year of cultivation, Average rainfall (in mm per year), Pesticide usage (in tonnes), Average temperature (in °C)

#### Dataset:
yield_df.csv (Uploaded under notebooks)

#### Model Training and Results
Various machine learning regression algorithms are applied, and the performance is evaluated based on metrics like Mean Squared Error (MSE).
The results of the models used are as follows:
1. Linear Regression
Mean Squared Error : 80852.08620158922
Score 0.09879301553673503

2. K Neighbors Regressor
Mean Squared Error : 55183.1146293406
Score 0.5801883304861266

3. Decision Tree Regressor
Mean Squared Error : 13211.190235825037
Score 0.9759383181169221

4. Random Forest Regressor
Mean Squared Error : 10135.46523142438
Score 0.9858378410451943

5. Gradient Boosting Regressor
Mean Squared Error : 34773.822585474634
Score 0.833295640875001

6. XGB Regressor
Mean Squared Error : 13451.947664464684
Score 0.975053338957936Linear Regression
Mean Squared Error : 80852.08620158922
Score 0.09879301553673503

The Random Forest Regressor was found to have the lowest MSE, making it the most suitable model for crop yield prediction. This model was selected for deployment and future predictions.

### 5. Mushroom Edibility Prediction
Develop a machine learning model that predicts whether a mushroom is edible or not, depending on it's physical features and environment. The model takes various inputs regarding the physical characteristics of the mushroom and outputs if the mushroom is edible or poisonous.

#### Dataset:
mushrooms.csv

#### Model Training and Results
Five different models were trained on the dataset to predict mushroom edibility. The accuracy of each model are as follows:

1. Logistic Regression: 0.94
2. Decision Tree Classifier: 1.0
3. K Nearest Neighbors: 0.99
4. Random Forest Classifier: 1.0
5. XGB Classifier: 1.0

The final model selected for deployment is the XGBoost Classifier as it can handle missing datas better than the other models.

## TechStack

- React
- Tailwind
- python - Flask
- Node
- MongoDB
- Express
- Machine Learning
- Deep Learning

<hr>

The API provides endpoints for various functionalities of the KisaanSetu platform. Below is a brief overview of the available endpoints:

### 1. Fertilizer Prediction
- **Endpoint:** `/api/fertilizer`
- **Method:** `POST`
- **Request Body:**
  ```json
  {
      "temperature": <number>,
      "humidity": <number>,
      "moisture": <number>,
      "soil_type": "<string>",
      "crop_type": "<string>",
      "nitrogen": <number>,
      "potassium": <number>,
      "phosphorus": <number>
  }
  ```
#### 2. Crop Prediction

- **Endpoint:** `/api/crop`
- **Method:** `POST`
- **Request Body:**
    ```json
    {
        "nitrogen": <number>,
        "phosphorus": <number>,
        "potassium": <number>,
        "temperature": <number>,
        "humidity": <number>,
        "ph_level": <number>,
        "rainfall": <number>
    }
    ```
#### 3. Soil Quality Prediction

**Endpoint:** `/api/soil-quality`  
**Method:** `POST`  

**Request Body:**
```json
{
    "nitrogen": <number>,
    "ph": <number>,
    "organic_carbon": <number>,
    "other_nutrients": {
        "phosphorus": <number>,
        "potassium": <number>
    }
}
```
#### 4. Yield Prediction

**Endpoint:** `/api/yield`  
**Method:** `POST`  

### Request Body:
```json
{
    "area": <number>,
    "crop_type": "<string>",
    "year": <number>,
    "rainfall": <number>,
    "pesticide_usage": <number>,
    "average_temperature": <number>
}
```
#### 5. Mushroom Edibility Prediction

**Endpoint:** `/api/mushroom`  
**Method:** `POST`  

### Request Body:
```json
{
    "cap_shape": "<string>",
    "cap_surface": "<string>",
    "cap_color": "<string>",
    "bruises": "<string>",
    "odor": "<string>",
    "gill_attachment": "<string>",
    "gill_spacing": "<string>",
    "gill_size": "<string>",
    "gill_color": "<string>",
    "stalk_shape": "<string>",
    "stalk_surface_above_ring": "<string>",
    "stalk_surface_below_ring": "<string>",
    "stalk_color_above_ring": "<string>",
    "stalk_color_below_ring": "<string>",
    "veil_type": "<string>",
    "veil_color": "<string>",
    "ring_number": "<string>",
    "ring_type": "<string>",
    "spore_print_color": "<string>",
    "population": "<string>",
    "habitat": "<string>"
}
```









