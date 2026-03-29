# ROAD-ACCIDENT-SEVERITY-PREDICTION

Machine Learning based Road Accident Severity Prediction system built using Django.  
The application predicts accident severity based on environmental and traffic-related features such as weather conditions, visibility, and road conditions.

## DataSet
Dataset used for training the model:

https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents

## Machine Learning Model
Random Forest was selected as the final model with an accuracy of 96%.

## Result

### Home Page
<img width="600" height="600" alt="image" src="https://github.com/user-attachments/assets/bc0ac1dd-1dac-46c2-afdb-b72a0d6773e7" />


The home page of the Accident Severity Prediction System provides an introduction to the web application.  
It explains that the system predicts the severity of road accidents using Machine Learning based on road and environmental conditions.  
Users can click the **"Start Prediction"** button to proceed to the prediction page where they can enter required parameters and obtain the predicted severity level.

### Prediction Page
<img width="500"  alt="image" src="https://github.com/user-attachments/assets/71baa514-ccc1-44cf-80bd-539d7d2b6f26" />


The prediction page allows users to enter related parameters required for severity prediction.  
Users can select their **"current location"** using the location button, which displays the location on an interactive map powered by **Leaflet and OpenStreetMap**.  

The form includes several input fields such as:

- Weather condition  
- Road surface condition  
- Light condition  
- Speed limit  
- Number of vehicles involved  
- Driver age  
- Driver gender  

After entering the required details, the user can click the **"Predict Severity"** button to generate the accident severity prediction using the trained **Machine Learning model (Random Forest)**.

### Result Page

<img width="600" height="600" alt="image" src="https://github.com/user-attachments/assets/8fe17404-b2e0-401c-a8f2-b1ed177550f4" />

This page displays the predicted accident severity after the user submits the required input parameters.  
The system processes the provided information using the trained **Random Forest Machine Learning model** and classifies the accident severity into three categories:

- **Minor**
- **Moderate**
- **Severe**

The predicted severity is highlighted on the screen to clearly indicate the result.  
Users can click the **Predict Again** button to return to the input form and perform another prediction with different parameters.
