import os 
import sys 
from src.exception import CustomException
from src.logger import logging
from dataclasses import dataclass

from src.utils import evaluate_model

from catboost import CatBoostRegressor
from sklearn.ensemble import (

AdaBoostRegressor,
GradientBoostingRegressor,
RandomForestRegressor,


)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from src.utils import save_object


@dataclass
class ModelTrainingConfig:
    trained_model_file_path=os.path.join("artifacts","model.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainingConfig()
    #creating Function for training
    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Splitting Training and test input Data ")
            x_train,y_train,x_test,y_test=(
               train_array[:,:-1],
               train_array[:,-1],
               test_array[:,:-1],
               test_array[:,-1],



            )
            
            models={
                     "Random Forest":RandomForestRegressor(),
                     "Adaboost regressor" : AdaBoostRegressor(),
                     "GradientBoostingRegressor" : GradientBoostingRegressor(),
                      "Linear Regression":LinearRegression(),
                      "KNN":KNeighborsRegressor(),
                      "CatBossting":CatBoostRegressor(verbose=False),
                      








            }

            model_report:dict=evaluate_model(x_train=x_train,y_train=y_train,x_test=x_test,y_test=y_test,models=models)
            best_model_score=max(sorted(model_report.values()))
            logging.info(best_model_score)
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model=models[best_model_name]
            print("best model name is :",best_model_name)

            if best_model_score <0.6:
                raise CustomException("No bets model found")
            logging.info("model was found best and traingi session has complete ")
            save_object(file_path=self.model_trainer_config.trained_model_file_path,
                        obj=best_model)
            predicted=best_model.predict(x_test)
            rsQuare=r2_score(y_test,predicted)
            logging.info(rsQuare)
            return rsQuare
             
        except Exception as e :
            CustomException(e,sys)
            




           