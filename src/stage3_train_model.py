from numpy import mod, split
from sklearn.linear_model import ElasticNet
import argparse
from src.utils.all_utils import read_yaml,create_directory,save_local_df
import os
import pandas as pd
from sklearn.linear_model import ElasticNet
import joblib

def trainModel(config_path,params_path):
    config = read_yaml(config_path)
    params = read_yaml(params_path)
    # Storing Local paths
    artifacts_dir = config['artifacts']['artifacts_dir']
    raw_local_dir = config['artifacts']['raw_local_dir']
    split_data_dir = config['artifacts']['split_data_dir']
    train = config['artifacts']['train']
    trainDataPath = os.path.join(artifacts_dir,split_data_dir,train)
    trainData = pd.read_csv(trainDataPath)
    # calling models parameters
    alpha = params['model_params']['ElasticNet']['alpha']
    l1_ratio = params['model_params']['ElasticNet']['l1_ratio']
    x,y = trainData.drop('quality',axis=1),trainData.quality
    model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio)
    model.fit(x,y)
    #Saving the model
    model_dir = config['artifacts']['model_dir']
    model_file_name = config['artifacts']['model_filename']
    model_dir = os.path.join(artifacts_dir, model_dir)
    create_directory([model_dir])
    #create_directory([os.path.join(artifacts_dir,model_dir)])
    model_path= os.path.join(model_dir,model_file_name)
    #saving the model
    joblib.dump(model, model_path)
    print(model_file_name)
    print(model_dir)
    return ''

if __name__ == '__main__':
    arg = argparse.ArgumentParser()
    arg.add_argument('--config','-c',default='config/config.yaml')
    arg.add_argument('--parms','-p',default='params.yaml')
args = arg.parse_args()
print(args)
print(args.config)
print(args.parms)
trainModel(args.config,args.parms)
