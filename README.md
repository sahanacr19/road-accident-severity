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
<img width="600" height="532" alt="image" src="https://github.com/user-attachments/assets/bc0ac1dd-1dac-46c2-afdb-b72a0d6773e7" />


The home page of the Accident Severity Prediction System provides an introduction to the web application.  
It explains that the system predicts the severity of road accidents using Machine Learning based on road and environmental conditions.  
Users can click the **"Start Prediction"** button to proceed to the prediction page where they can enter required parameters and obtain the predicted severity level.

### User Page
<img width="461" height="801" alt="image" src="https://github.com/user-attachments/assets/7824c187-39c8-406a-95d7-5e37d49ee649" />

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
