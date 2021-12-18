import argparse
from os import sep
import pandas as pd
from src.utils.all_utils import read_yaml,create_directory
import os

def get_data(config_path):
    config = read_yaml(config_path)
    remote_path = config['data_source']
    data = pd.read_csv(remote_path,sep=';')
    # save dataset in the local directory
    # create path to directory: artifacts/raw_local_dir/data.csv
    artifacts_dir = config["artifacts"]['artifacts_dir']
    raw_local_dir = config["artifacts"]['raw_local_dir']
    raw_local_file = config["artifacts"]['raw_local_file']
    #print(artifacts_dir,raw_local_dir,raw_local_file)
    raw_local_dir_path = os.path.join(artifacts_dir, raw_local_dir)
    
    create_directory(dirs = [raw_local_dir_path])
    raw_local_file_path = os.path.join(raw_local_dir_path, raw_local_file)
    print(raw_local_file_path)
    data.to_csv(raw_local_file_path, sep=",", index=False)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', "-c", default="config/config.yaml")

args = parser.parse_args()
get_data(config_path = args.config)
