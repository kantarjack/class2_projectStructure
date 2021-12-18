import joblib
import argparse
import os
import pandas as pd
import numpy as np
from src.utils.all_utils import save_reports
from utils.all_utils import create_directory,read_yaml,save_local_df
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def evaluate_metrics(actual_values, predicted_values):
    rmse = np.sqrt(mean_squared_error(actual_values, predicted_values))
    mae = mean_absolute_error(actual_values, predicted_values)
    r2 = r2_score(actual_values, predicted_values)
    return rmse, mae, r2

def evaluate(config_path,params_path):
    config = read_yaml(config_path) 
    params =  read_yaml(params_path)
    artifact_dir = config['artifacts']['artifacts_dir']
    split_data_dir = config['artifacts']['split_data_dir']
    test = config['artifacts']['test']
    model_dir = config['artifacts']['model_dir']
    model_filename = config['artifacts']['model_filename']
    model_path = os.path.join(artifact_dir,model_dir,model_filename)
    model = joblib.load(model_path)
    print('Model Loaded')

    
    #Test the model
    #load test data
    test_data_path = os.path.join(artifact_dir,split_data_dir,test)
    test_data = pd.read_csv(test_data_path)
    x_test,y_test = test_data.drop('quality',axis=1),test_data.quality

    #Model predictions
    predicted_values = model.predict(x_test)

    rmse, mae, r2 = evaluate_metrics(y_test,predicted_values)

    ## Saving the model
    report = {
        'rmse':rmse,
        'mae':mae,
        'r2':r2
    }
    print(report)
    reports_dir = config['artifacts']['reports_dir']
    scores_file = config['artifacts']['scores']
    reports_dir_path= os.path.join(artifact_dir,reports_dir)
    print(reports_dir_path)
    create_directory([reports_dir_path])
    reports_file_path = os.path.join(reports_dir_path,scores_file)
    save_reports(report,reports_file_path)


if __name__ == '__main__':
    arg = argparse.ArgumentParser()
    arg.add_argument('--config','-c',default='config/config.yaml')
    arg.add_argument('--params','-p',default = 'params.yaml')
args = arg.parse_args()    
evaluate(args.config,args.params)


     