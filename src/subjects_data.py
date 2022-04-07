import scipy
import os
import numpy as np
import pandas as pd
from tqdm.notebook import tqdm

def unpack_mat_struct(struct):
    """Recursively unpack scipy.io.matlab.mio5_params.mat_struct types in a dict. 
    Unpacking is done by the Python's built-in command vars()

    Args:
        struct (dict): result of scipy.io.loadmat function
    """
    for k in struct.keys():
        if isinstance(struct[k], scipy.io.matlab.mio5_params.mat_struct):
            struct[k] = vars(struct[k])
            unpack_mat_struct(struct[k])
        elif isinstance(struct[k], dict):
            unpack_mat_struct(struct[k])

def read_mat_file(filename):
    """Reads .mat file into a dictionary

    Args:
        path (str): path to mat file
    """
    data = scipy.io.loadmat(filename, struct_as_record=False, squeeze_me=True)
    unpack_mat_struct(data)
    return data

def read_subject_to_dict(subject_dirname):
    """Reads subject folder of .mat and .txt files into a dict

    Args:
        name (str): name of subject. Assuming files located in 'args.path/name/'
    """
    files = os.listdir(subject_dirname)
    data = {"name":os.path.basename(subject_dirname)}
    for f in files:
        abs_filename = os.path.join(subject_dirname, f)
        
        if f.endswith("mat"):
            struct = scipy.io.loadmat(abs_filename, struct_as_record=False, squeeze_me=True)
            unpack_mat_struct(struct)
            data.update(struct)
        
        elif f.endswith("txt"): # txt file consists only a number indicating a bad trial
            with open(abs_filename, 'r') as txt_file:
                data["bad_trial"] = txt_file.read() # reading lazely. processing will be done elsewhere
                
    return data


def preprocess_subject_dict(subject_dict):
    """
    Handle bad trials, including shape missmatches, remove keys with leading __
    """
    if "bad_trial" in subject_dict.keys():
        for k in ["probPerTrial", "AUCperTrial"]:
            subject_dict[k] = np.insert(subject_dict[k], int(subject_dict["bad_trial"])-1, np.nan)

def extract_subject_dict_to_df(subject_dict):
    df = pd.DataFrame({
    "block": subject_dict["sequenceData"]["seq1"],
    "prob": subject_dict["probPerTrial"],
    "auc": subject_dict["AUCperTrial"]
    })

    df["name"] = subject_dict["name"]
    df["trial"] = df.index
    df["N"] = subject_dict["featureInfo"]["modelParams"]["pastLength"]
    beta_ind = subject_dict["featureInfo"]["modelParams"]["betaInd"] - 1
    df["beta_ind"] = beta_ind
    df["beta"] = subject_dict["featureInfo"]["modelParams"]["allBeta"][beta_ind]
    
    df["normal_factor"] = subject_dict["featureInfo"]["plotData"]["meanDiffSignalAUC"]
    return df

def read_all(path):
    """Reads all subjects data from 'path'. 
    'path' is assumed to have folders with subject names.

    Args:
        path (str): path to folder that contains folders, each of these folders contains .mat and .txt files
    """
    df_subject_list = []
    for name in tqdm(os.listdir(path)):
        data_dict = read_subject_to_dict(os.path.join(path, name))
        preprocess_subject_dict(data_dict)
        try:
            df = extract_subject_dict_to_df(data_dict)
            df_subject_list.append(df)
        except Exception as e:
            print(f"Exception in {name}:\n{str(e)}")
    
    return pd.concat(df_subject_list)
        