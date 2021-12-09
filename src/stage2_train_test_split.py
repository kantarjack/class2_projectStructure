from os import read
import os
from utils.all_utils import create_directory,read_yaml,save_local_df
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split

def split_save(config_path,params_path):
    config = read_yaml(config_path)
    params = read_yaml(params_path)
    print(config)  
    artifacts_dir = config["artifacts"]['artifacts_dir']
    raw_local_dir = config["artifacts"]['raw_local_dir']
    raw_local_file = config["artifacts"]['raw_local_file']
    raw_local_file_path = os.path.join(artifacts_dir,raw_local_dir,raw_local_file)
    data = pd.read_csv(raw_local_file_path)

    #Spitting the data
    split_ratio = params['base']['test_size']
    random_state = params['base']['random_state']
    print(split_ratio)
    train,test = train_test_split(data,test_size=.5, random_state=random_state)
    print(test.head())
    #saving to local
    split_data_dir = config["artifacts"]['split_data_dir']
    create_directory([os.path.join(artifacts_dir,split_data_dir)])

    train_data_file_name = config["artifacts"]['train']
    test_data_file_name = config["artifacts"]['test']

    
    train_data_path = os.path.join(artifacts_dir,split_data_dir,train_data_file_name)
    test_data_path = os.path.join(artifacts_dir,split_data_dir,test_data_file_name)
    
    for i,j in ((train,train_data_path),(test,test_data_path)):
        save_local_df(i,j)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config","-c",default='config/config.yaml')
    parser.add_argument("--params","-p",default='params.yaml')

args = parser.parse_args()
split_save(args.config,args.params)

