import pandas as pd 
import os 
import sys
from src.exception import CustomException
from src.logger import logging

from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig
from src.components.model_trainer import ModelTrainer,ModelTrainingConfig










@dataclass
class DataIngestionConfig:
    #these are the input givven to data ingestion componenet
    train_data_path:str=os.path.join('artifacts',"train.csv") #isme ham trainign data ssave kre ge 
    test_data_path:str=os.path.join('artifacts',"test.csv")
    raw_data_path:str=os.path.join('artifacts',"raw.csv")
class DataIngestion:
    def __init__(self) :
       self.ingestion_config=DataIngestionConfig() #isme oper wali class se data paths arhay hain
    
    
    def initiate_data_ingestion(self):
        logging.info("enter the Data Ingestion")#wrting the logs 
        try:#reading Data
            df=pd.read_csv('notebook\data\stud.csv')
            logging.info("read the Data as data frame  ")
            

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)#creating the training directrory while loadaing from the main directory


            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)
          #seaving or writing to Data 
            logging.info("train test split intitaed ")

            train_set,test_set=train_test_split(df,test_size=0.2,random_state=42)
            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)
            logging.info("ingestion Of Data is complete ")

            return(
                 self.ingestion_config.train_data_path,
                 self.ingestion_config.test_data_path



            )
        except Exception as e:
            raise CustomException(e,sys) 

#checking if it is working well

if __name__=="__main__":
    obj=DataIngestion()
    train_data,test_data=obj.initiate_data_ingestion()
    data_transfromation=DataTransformation()
    train_arr,test_arr,_ =data_transfromation.initiate_data_transformation(train_data,test_data)
    model_trainer=ModelTrainer()
    print(model_trainer.initiate_model_trainer(train_arr,test_arr))
