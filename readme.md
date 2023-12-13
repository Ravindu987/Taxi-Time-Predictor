# Taxi Time Predictor


Prediction of taxi time duration of New York City given certain data

Done as a learning project for industrial-level ML and MLOPs.

Done using the Kaggle Dataset [New York City Taxi Trip Duration](https://www.kaggle.com/competitions/nyc-taxi-trip-duration/overview)


#### Machine Learning Aspect

Used technologies
- Pandas
- Numpy
- Scikit-Learn
- XGBoost
- PyCaret

The dataset was analyzed, preprocessed and then used to train a model.
Pycaret was used to train the dataset using different model types and idenitfy the best performing model.
hyperopt was used to optimize hyperparameters.
The final XGBoost model showed 229.08 rmse.

The experiments can be viewd on [Dagshub repo](https://dagshub.com/Ravindu987/Taxi_time_prediction)


#### MLOps Aspect

Used technologies
- Dataprep: Data Analysis
- PyCaret: Model development
- DVC: Versioning datasets
- Dagshub: Repo for data and experiments
- MLflow: Run and track experiments
- BentoML: Build APIs for model


#### Front-end Developed Using React