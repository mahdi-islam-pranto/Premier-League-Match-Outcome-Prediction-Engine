import sys
import os
from src.logger import logging
from src.exception import CustomException
import pickle
import dill
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

# utility function to save the preprocessor object (ml model object after training -> pickle file) to the specified path
def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
            
    except Exception as e:
        logging.info("Exception occurred while saving object")
        raise CustomException(e, sys)
    
    
def evaluate_models(X_train, y_train, X_test, y_test, models, params):
    
    try:
        report = {}
        
        for i in range(len(models)):
            # getting the model name from the models dictionary
            model = list(models.values())[i]
            # getting the parameters for the model from the params dictionary
            parameters = params[list(models.keys())[i]]
            
            # creating the GridSearchCV object for the model and parameters
            gs = GridSearchCV(model,parameters,cv=3)
            gs.fit(X_train, y_train)
            # setting the model parameters to the best parameters found by GridSearchCV
            model.set_params(**gs.best_params_)
            
            # Train model
            model.fit(X_train, y_train)
            # predicting train and test data
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            
            # calculating r2 score for train and test data
            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)
            
            report[list(models.keys())[i]] = test_model_score
            
        return report
            
    except Exception as e:
        logging.info("Exception occurred while evaluating models")
        raise CustomException(e, sys)
    

# utility function to load the preprocessor object or model object (ml model object after training -> pickle file) from the specified path
def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    