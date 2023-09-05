import pandas as pd
import numpy as np
import tqdm
import pickle
import ast
import datetime
import logging
def showwarning(message, category, filename, lineno, file=None, line=None):
    pass
import warnings
warnings.showwarning = showwarning
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import normalize
from sklearn.ensemble import GradientBoostingClassifier 
from matplotlib import pyplot as plt
from fields_and_encodings import Fields, Encodings
from type_functions import TypeFunctions, ColumnCleaner    

from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from tabulate import tabulate

# hyper params
# hyper params


def output_to_log_and_terminal(output, level=logging.INFO):
    print(output)
    logging.log(level, output)


def table_columns(field_names):
    positive = pd.read_csv('csv/instance_0_positive.csv',nrows=1)
    selected_cols = [
        col 
        for field in field_names 
        for col in positive.columns 
        if col.startswith(str(field) + "-0")
        # we only wan't exacly the field asked from instance 0
    ]
    return selected_cols

def create_mixed_dataset(selected_cols):
    row_num = 0
    positive = pd.read_csv('csv/instance_0_positive.csv',usecols=selected_cols, chunksize=1000,low_memory=False)
    negative = pd.read_csv('csv/all_instance_negative.csv',usecols=selected_cols, chunksize=1000,low_memory=False)

    cols = positive.get_chunk(0).columns
    print(len(cols))
    s_chunk = pd.DataFrame(columns=cols)
    s_chunk.to_csv('csv/shuffled_dataset_first_itter.csv',index=False,mode='w')
    for p_chunk, n_chunk in tqdm.tqdm(zip(positive,negative), total=11):
        s_chunk = pd.concat([p_chunk,n_chunk]).sample(frac=1)
        row_num += s_chunk.shape[0]
        s_chunk[cols].to_csv('csv/shuffled_dataset_first_itter.csv',mode='a',header=False,index=False)
    del positive,negative, p_chunk, s_chunk, n_chunk
    return row_num


def map_fields_to_cleaners(field_names, field_types, field_encodings, selected_cols, enc_dict):
    field_groups = {}
    for f, t, e in tqdm.tqdm(zip(field_names,field_types,field_encodings), total=len(field_names)):
        # we wan't to avoid changing the final label
        if f != 6152:
            res = []
            f = str(f)
            for col in selected_cols:
                if col.startswith(f + '-'):
                    res.append(col)
            field_groups[f] = ColumnCleaner(res, t, enc_dict.get(e, []), f)
    return list(field_groups.values())

def activate_cleaners(dataset, cleaners, selected_cols, train_size):
    ######
    # Added the 6152 here 
    # there should be a better way to add the 6152 but fuck it
    y_cols = []
    for col in selected_cols:
            if col.startswith('6152-'):
                y_cols.append(col)
    ######

    # copy chunks while cleaning dtypes
    for i,chunk in enumerate(dataset):
        res_cols = []
        for cleaner in tqdm.tqdm(cleaners,desc=f"chunk {i+1}/1"):
            clean_data = cleaner.fit_transform(chunk[cleaner.column_names], train_size)
            chunk.drop(columns=cleaner.column_names, inplace=True)
            if not clean_data.empty:
                res_cols.append(clean_data)

        res_cols.append(chunk[y_cols])
        chunk = pd.concat(res_cols,axis=1)
        chunk.to_csv('csv/shuffled_dataset_clean.csv',mode='w',header=True,index=False)
    return chunk.dtypes



def prepare_data(exp_name):
    # get desired field list
    field_names, field_types, field_encodings = Fields.load_desired_fields()

    #output_to_log_and_terminal("loaded desired fields")

    # build final field list including instances and arrays...
    selected_cols = table_columns(field_names)

    #output_to_log_and_terminal("loaded resulting table columns")

    # build final dataset from positive and negative with only needed columns
    row_num = create_mixed_dataset(selected_cols)

    # claculate train and test set seperation point
    # the dataset is already shuffled so the seperation point is constant
    train_size = row_num * 7 // 10

    """ output_to_log_and_terminal(
        "created mixed dataset, row count: " + 
        str(row_num) + 
        " train size: " + 
        str(train_size)
    ) """

    # create a clean dataset with clean datatypes
    dataset = pd.read_csv('csv/shuffled_dataset_first_itter.csv',chunksize=row_num, low_memory=False)

    # build dict of encoding values
    enc_dict = Encodings.create_category_dictionary()

    # build final field to type dictionary
    cleaners = map_fields_to_cleaners(field_names, field_types, field_encodings, selected_cols, enc_dict)

    #output_to_log_and_terminal("done with type matching")

    type_list = activate_cleaners(dataset,cleaners,selected_cols,train_size)

    #output_to_log_and_terminal("shuffled_dataset_clean.csv is created")

    # if all works this will print only number and float dtypes
    #print(np.unique(type_list).tolist())
    return cleaners

def load_data(exp_name):
    # seperate x and y fields
    dataset = pd.read_csv('csv/shuffled_dataset_clean.csv', nrows=1)

    # 6152 is the field we want to predict
    y_cols = [col for col in dataset.columns if col.startswith("6152-")]
    x_cols = [col for col in dataset.columns if not col.startswith("6152-")]
    
    dataset = pd.read_csv('csv/shuffled_dataset_clean.csv', usecols=x_cols)
    X = dataset.to_numpy()
    dataset = pd.read_csv('csv/shuffled_dataset_clean.csv', usecols=y_cols)
    Y = np.sum(dataset[y_cols].to_numpy()==5,axis=-1)!=0
    
    cleaner_list = np.load("logs/"+exp_name+"_cleaner_list.npy", allow_pickle=True)

    return X,Y, x_cols, cleaner_list

def experiment(exp_name,remake_dataset, cluster:bool):
    #####
    # basically just added a logging system to keep my sanity
    # add a folder named logs otherwise this will crash
    #####

    logging.basicConfig(filename='logs/'+ exp_name + "__" + str(datetime.datetime.now())
                        +".log", encoding = 'utf-8', level=logging.DEBUG)
    
    output_to_log_and_terminal("start: " + str(datetime.datetime.now()))

    cleaner_list = []
    if remake_dataset:
        #output_to_log_and_terminal("remake dataset")
        cleaner_list = prepare_data(exp_name)
        np.save("logs/"+exp_name+"_cleaner_list.npy",cleaner_list,allow_pickle=True)
    X,Y, x_cols, cleaner_list = load_data(exp_name)


    ### clustering
    # dbscan = DBSCAN(eps=0.5, min_samples=5)  
    # cluster_labels = dbscan.fit_predict(X)

    train_limit = X.shape[0]*7//10

    if cluster:
        Xtrain, Ytrain = clusterData(X[:train_limit], Y[:train_limit], 3000)
    else:
        Xtrain, Ytrain = X[:train_limit], Y[:train_limit]

    # Xtrain = X[:train_limit]
    # Ytrain = Y[:train_limit]

    output_to_log_and_terminal("loaded data")
    clf = LogisticRegression()
    output_to_log_and_terminal("before fit")
    clf.fit(Xtrain, Ytrain)
    output_to_log_and_terminal("after fit")

    test_score = clf.score(X[train_limit:], Y[train_limit:])
    train_score = clf.score(Xtrain, Ytrain)
    output_to_log_and_terminal("test score :" + str(test_score))
    output_to_log_and_terminal("train score :" + str(train_score))
    classification_report_pretty_print(Y[train_limit:], clf.predict(X[train_limit:]))
    output_to_log_and_terminal("end: " + str(datetime.datetime.now()))

    s = pickle.dumps(clf)
    with open('logs/'+ exp_name+".pkl",'wb') as f_out:
        f_out.write(s)

    return clf, x_cols


def experimentXgBoost(exp_name,remake_dataset, cluster:bool):
    #####
    # basically just added a logging system to keep my sanity
    # add a folder named logs otherwise this will crash
    #####

    logging.basicConfig(filename='logs/'+ exp_name + "__" + str(datetime.datetime.now())
                        +".log", encoding = 'utf-8', level=logging.DEBUG)
    
    output_to_log_and_terminal("start: " + str(datetime.datetime.now()))

    cleaner_list = []
    if remake_dataset:
        output_to_log_and_terminal("remake dataset")
        cleaner_list = prepare_data(exp_name)
        np.save("logs/"+exp_name+"_cleaner_list",cleaner_list,allow_pickle=True)
    X,Y, x_cols, cleaner_list = load_data(exp_name)

    train_limit = X.shape[0]*7//10

    if cluster:
        Xtrain, Ytrain = clusterData(X[:train_limit], Y[:train_limit], 3000)
    else:
        Xtrain, Ytrain = X[:train_limit], Y[:train_limit]
    # Xtrain = X[:train_limit]
    # Ytrain = Y[:train_limit]
    output_to_log_and_terminal("loaded data")
    clf = GradientBoostingClassifier(n_iter_no_change=10)
    output_to_log_and_terminal("before fit")
    clf.fit(Xtrain, Ytrain)
    output_to_log_and_terminal("after fit")

    test_score = clf.score(X[train_limit:], Y[train_limit:])
    train_score = clf.score(Xtrain, Ytrain)
    output_to_log_and_terminal("test score :" + str(test_score))
    output_to_log_and_terminal("train score :" + str(train_score))
    classification_report_pretty_print(Y[train_limit:], clf.predict(X[train_limit:]))
    output_to_log_and_terminal("end: " + str(datetime.datetime.now()))

    s = pickle.dumps(clf)
    with open('logs/'+ exp_name+".pkl",'wb') as f_out:
        f_out.write(s)

    return clf, x_cols

def classification_report_pretty_print(Y_true:np.ndarray, Y_pred:np.ndarray):
    # Compute the classification report
    report = classification_report(Y_true, Y_pred, target_names=['healthy', 'sick'])

    # Split the report into lines and format it as a table
    report_table = [line.split() for line in report.split('\n')[2:-5]]

    # Print the nicely formatted table
    output_to_log_and_terminal(tabulate(report_table, headers=['Class', 'Precision', 'Recall', 'F1-Score', 'Support'], tablefmt='grid'))

def plot_fields(exp_name, x_cols, weights, important_weight_num, remove_zero:bool=False):
    data = sorted(zip(x_cols, weights),key=lambda x: x[1],reverse=True)
    sorted_x_cols = [x[0] for x in data]
    weights = [x[1] for x in data]

    if remove_zero:
        for col, weight in data:
            if weight == 0:
                dotIndx = col.find('.')
                dashIndx = col.find('-')
                if dashIndx < dotIndx and dashIndx != -1:
                    Fields.add_field_to_unwanted_fields_file(int(col[:dashIndx]),"Zero weight")
                else:
                    Encodings.add_values_to_unwanted_valuse_file(int(col[:dotIndx]),col[dotIndx+1:],"Zero weight")

    #fig = plt.figure(figsize=(40,40))
    plt.barh(
        sorted_x_cols[:important_weight_num],
        np.log(
            1+
            np.abs(
                weights[:important_weight_num]
            )
        ),height=0.5)
    plt.xlabel("lan(1+|weight|)")
    plt.ylabel("field")
    plt.savefig('logs/'+ exp_name + "__" + str(datetime.datetime.now())
                + "__" +'weights.png',bbox_inches = 'tight')
    plt.show()

def plot_XGBoost(exp_name, x_cols, weights, important_weight_num, remove_zero:bool=False):
    data = sorted(zip(x_cols, weights),key=lambda x: x[1],reverse=True)
    sorted_x_cols = [x[0] for x in data]
    weights = [x[1] for x in data]

    if remove_zero:
        for col, weight in data:
            if weight == 0:
                dotIndx = col.find('.')
                dashIndx = col.find('-')
                if dashIndx < dotIndx and dashIndx != -1:
                    Fields.add_field_to_unwanted_fields_file(int(col[:dashIndx]),"Zero weight")
                else:
                    Encodings.add_values_to_unwanted_valuse_file(int(col[:dotIndx]),col[dotIndx+1:],"Zero weight")
    # fig = plt.figure(figsize=(40,40))
    plt.barh(
        sorted_x_cols[:important_weight_num],
        weights[:important_weight_num],
        height=0.5)
    plt.xlabel("weight")
    plt.ylabel("field")
    plt.savefig('logs/'+ exp_name + "__" + str(datetime.datetime.now())
                + "__" +'weights.png',bbox_inches = 'tight')
    plt.show()



def clusterData(X, y, k):

    # Step 1: Cluster the Data
    kmeans = KMeans(n_clusters=k)
    cluster_assignments = kmeans.fit_predict(X)

    # Step 2: Extract Cluster Centers (Centroids)
    cluster_centers = kmeans.cluster_centers_

    # Step 3: Create New Examples (Centers)
    centers = cluster_centers

    # Step 4: Label the New Examples
    examples=[]
    new_example_labels = []
    for i, center in enumerate(centers):
        # Calculate the distances from the center to all data points in the cluster
        distances = np.linalg.norm(X[cluster_assignments == i] - center, axis=1)
        min_example_index = distances.argmin()
        examples.append(X[cluster_assignments == i][min_example_index])
        new_example_labels.append(y[cluster_assignments == i][min_example_index])
    
    return np.array(examples), new_example_labels
