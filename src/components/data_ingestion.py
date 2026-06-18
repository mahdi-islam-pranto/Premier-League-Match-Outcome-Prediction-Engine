import os
import sys
import pandas as pd
from dataclasses import dataclass
from pathlib import Path
# ensure project root is on sys.path when running this file directly
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))
from src.logger import logging
from src.exception import CustomException

# class for data ingestion configuration, where we will specify the path to store the train, test and raw data
@dataclass
class DataIngestionConfig:
    raw_data_path: str = os.path.join('artifacts', 'raw.csv')
    train_data_path: str = os.path.join('artifacts', 'train.csv')
    val_data_path: str = os.path.join('artifacts', 'val.csv')
    test_data_path: str = os.path.join('artifacts', 'test.csv')
    
class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
        
    def get_season(self, date):
        # EPL season runs Aug -> May. Aug-Dec belongs to season starting that year.
        year = date.year
        return f"{year}-{year+1}" if date.month >= 7 else f"{year-1}-{year}"
    
    def initiate_data_ingestion(self):
        logging.info("Starting data ingestion process.")
        
        try:
            logging.info("Reading merged 10-season EPL dataset")
            
            df = pd.read_csv('notebooks/datasets/pl-matches-dataset-16-26.csv')
            
            # convert match date to datetime and sort by date, then extract season info
            df['match_date'] = pd.to_datetime(df['match_date'])
            df = df.sort_values('match_date').reset_index(drop=True)
            df['season'] = df['match_date'].apply(self.get_season)
            # save the raw data to the artifacts folder
            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            logging.info("Raw data saved successfully.")
            
            logging.info("Splitting chronologically by season (no shuffling)")
            seasons_sorted = sorted(df['season'].unique())
            train_seasons = seasons_sorted[:-2]   # first 8 seasons
            val_season    = seasons_sorted[-2]    # 9th season
            test_season   = seasons_sorted[-1]    # most recent season
            
            # separate data by train, val and test
            train_set = df[df['season'].isin(train_seasons)]
            val_set   = df[df['season'] == val_season]
            test_set  = df[df['season'] == test_season]
            
            # save the all the datasets to artifacts
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            val_set.to_csv(self.ingestion_config.val_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)
            
            logging.info(f"Train: {len(train_set)} | Val: {len(val_set)} | Test: {len(test_set)} matches")

            
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.val_data_path,
                self.ingestion_config.test_data_path
            )
            
        except Exception as e:
            logging.error(f"Error reading dataset: {e}")
            raise CustomException(e, sys)
        
        
if __name__ == "__main__":
    obj = DataIngestion()
    obj.initiate_data_ingestion()