import pandas as pd
import numpy as np
import tqdm
import pickle
import datetime
import logging
def showwarning(message, category, filename, lineno, file=None, line=None):
    pass
import warnings
warnings.showwarning = showwarning
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier 
from matplotlib import pyplot as plt
from fields_and_encodings import Fields, Encodings
from type_functions import TypeFunctions, ColumnCleaner    

from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from tabulate import tabulate
from sklearn.metrics import f1_score
import itertools

# hyper params
# hyper params


def output_to_log_and_terminal(output, block_output:bool, level=logging.INFO):
    if not block_output:
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



def prepare_data(exp_name, block_output:bool):
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

def load_data(exp_name, block_output:bool=False):
    # seperate x and y fields
    dataset = pd.read_csv('csv/shuffled_dataset_clean.csv', nrows=1)

    # 6152 is the field we want to predict
    y_cols = [col for col in dataset.columns if col.startswith("6152-")]
    x_cols = [col for col in dataset.columns if not col.startswith("6152-")]

    x_cols = filter_final_fields(x_cols, block_output)
    
    dataset = pd.read_csv('csv/shuffled_dataset_clean.csv', usecols=x_cols)
    X = dataset.to_numpy()
    dataset = pd.read_csv('csv/shuffled_dataset_clean.csv', usecols=y_cols)
    Y = np.sum(dataset[y_cols].to_numpy()==5,axis=-1)!=0
    
    cleaner_list = np.load("logs/"+exp_name+"_cleaner_list.npy", allow_pickle=True)

    return X,Y, x_cols, cleaner_list


def filter_final_fields(x_cols, block_output:bool=False):
    final_fields = x_cols
    try:
        final_fields = pd.read_csv('csv/keep_fields.csv')['final_field_name'].astype(str).tolist()
        final_fields = set(final_fields).intersection(set(x_cols))
        if len(final_fields) == 0:
            final_fields = x_cols
        output_to_log_and_terminal("filtered field list, old size: " + str(len(x_cols)) + ", new size: " + str(len(final_fields)), block_output)
    except:
        pass
    return final_fields

def experiment(exp_name,remake_dataset, cluster:bool, block_output:bool=False):
    #####
    # basically just added a logging system to keep my sanity
    # add a folder named logs otherwise this will crash
    #####

    logging.basicConfig(filename='logs/'+ exp_name + "__" + str(datetime.datetime.now())
                        +".log", encoding = 'utf-8', level=logging.DEBUG)
    
    output_to_log_and_terminal("start: " + str(datetime.datetime.now()), block_output)

    cleaner_list = []
    if remake_dataset:
        output_to_log_and_terminal("remake dataset", block_output)
        cleaner_list = prepare_data(exp_name,block_output)
        np.save("logs/"+exp_name+"_cleaner_list.npy",cleaner_list,allow_pickle=True)
    X,Y, x_cols, cleaner_list = load_data(exp_name, block_output)

    train_limit = X.shape[0]*7//10

    if cluster:
        Xtrain, Ytrain = clusterData(X[:train_limit], Y[:train_limit], 3000)
    else:
        Xtrain, Ytrain = X[:train_limit], Y[:train_limit]

    # Xtrain = X[:train_limit]
    # Ytrain = Y[:train_limit]

    output_to_log_and_terminal("loaded data", block_output)
    clf = LogisticRegression()
    output_to_log_and_terminal("before fit", block_output)
    clf.fit(Xtrain, Ytrain)
    output_to_log_and_terminal("after fit", block_output)

    Ypred = clf.predict(X[train_limit:])
    test_score = clf.score(X[train_limit:], Y[train_limit:])
    train_score = clf.score(Xtrain, Ytrain)
    output_to_log_and_terminal("test score :" + str(test_score), block_output)
    output_to_log_and_terminal("train score :" + str(train_score), block_output)
    classification_report_pretty_print(Y[train_limit:], Ypred, block_output)
    show_confusion_matrix(Y[train_limit:], Ypred)
    output_to_log_and_terminal("end: " + str(datetime.datetime.now()), block_output)

    return clf, x_cols


def experimentSVM(exp_name,remake_dataset, cluster:bool, block_output:bool=False, kernel='rbf'):
    #####
    # basically just added a logging system to keep my sanity
    # add a folder named logs otherwise this will crash
    #####

    logging.basicConfig(filename='logs/'+ exp_name + "__" + str(datetime.datetime.now())
                        +".log", encoding = 'utf-8', level=logging.DEBUG)
    
    output_to_log_and_terminal("start: " + str(datetime.datetime.now()), block_output)

    cleaner_list = []
    if remake_dataset:
        output_to_log_and_terminal("remake dataset", block_output)
        cleaner_list = prepare_data(exp_name,block_output)
        np.save("logs/"+exp_name+"_cleaner_list.npy",cleaner_list,allow_pickle=True)
    X,Y, x_cols, cleaner_list = load_data(exp_name)

    train_limit = X.shape[0]*7//10

    if cluster:
        Xtrain, Ytrain = clusterData(X[:train_limit], Y[:train_limit], 3000)
    else:
        Xtrain, Ytrain = X[:train_limit], Y[:train_limit]

    # Xtrain = X[:train_limit]
    # Ytrain = Y[:train_limit]

    output_to_log_and_terminal("loaded data", block_output)

    clf = SVC(kernel=kernel)
    output_to_log_and_terminal("before fit", block_output)
    clf.fit(Xtrain, Ytrain)
    output_to_log_and_terminal("after fit", block_output)
    Ypred = clf.predict(X[train_limit:])
    test_score = clf.score(X[train_limit:], Y[train_limit:])
    train_score = clf.score(Xtrain, Ytrain)
    output_to_log_and_terminal("test score :" + str(test_score), block_output)
    output_to_log_and_terminal("train score :" + str(train_score), block_output)
    classification_report_pretty_print(Y[train_limit:], Ypred, block_output)
    show_confusion_matrix(Y[train_limit:], Ypred)
    output_to_log_and_terminal("end: " + str(datetime.datetime.now()), block_output)



    s = pickle.dumps(clf)
    with open('logs/'+ exp_name+".pkl",'wb') as f_out:
        f_out.write(s)

    return clf, x_cols


def experimentXgBoost(exp_name,remake_dataset, cluster:bool, block_output:bool=False, algoParams:dict=None, ax:plt.Figure=None):
    #####
    # basically just added a logging system to keep my sanity
    # add a folder named logs otherwise this will crash
    #####

    logging.basicConfig(filename='logs/'+ exp_name + "__" + str(datetime.datetime.now())
                        +".log", encoding = 'utf-8', level=logging.DEBUG)
    
    output_to_log_and_terminal("start: " + str(datetime.datetime.now()), block_output)

    cleaner_list = []
    if remake_dataset:
        output_to_log_and_terminal("remake dataset", block_output)
        cleaner_list = prepare_data(exp_name, block_output)
        np.save("logs/"+exp_name+"_cleaner_list",cleaner_list,allow_pickle=True)
    X,Y, x_cols, cleaner_list = load_data(exp_name,block_output)

    train_limit = X.shape[0]*7//10

    if cluster:
        Xtrain, Ytrain = clusterData(X[:train_limit], Y[:train_limit], 3000)
    else:
        Xtrain, Ytrain = X[:train_limit], Y[:train_limit]
    # Xtrain = X[:train_limit]
    # Ytrain = Y[:train_limit]
    output_to_log_and_terminal("loaded data", block_output)
    if algoParams is None:
        clf = GradientBoostingClassifier()
    else:
        clf = GradientBoostingClassifier(**algoParams)
    output_to_log_and_terminal("before fit", block_output)
    clf.fit(Xtrain, Ytrain)
    output_to_log_and_terminal("after fit", block_output)

    test_score = clf.score(X[train_limit:], Y[train_limit:])
    train_score = clf.score(Xtrain, Ytrain)
    Ypred = clf.predict(X[train_limit:])
    output_to_log_and_terminal("test score :" + str(test_score), block_output)
    output_to_log_and_terminal("train score :" + str(train_score), block_output)
    f1 = classification_report_pretty_print(Y[train_limit:], Ypred, block_output)
    if ax is not None:
        plt.scatter(algoParams['n_estimators'], f1, color='red', marker='o')
        plt.annotate('depth: '+str(algoParams['max_depth']), (algoParams['n_estimators'], f1), textcoords="offset points", xytext=(10, 10), ha='center')
    show_confusion_matrix(Y[train_limit:], Ypred, block_output)
    output_to_log_and_terminal("end: " + str(datetime.datetime.now()), block_output)

    s = pickle.dumps(clf)
    with open('logs/'+ exp_name+".pkl",'wb') as f_out:
        f_out.write(s)

    return clf, x_cols

def classification_report_pretty_print(Y_true:np.ndarray, Y_pred:np.ndarray, block_output:bool):
    # Compute the classification report
    report = classification_report(Y_true, Y_pred, target_names=['healthy', 'sick'])
    # Split the report into lines and format it as a table
    report_table = [line.split() for line in report.split('\n')[2:-5]]

    # Print the nicely formatted table
    output_to_log_and_terminal(tabulate(report_table, headers=['Class', 'Precision', 'Recall', 'F1-Score', 'Support'], tablefmt='grid')
                               , block_output)
    return f1_score(Y_true, Y_pred)

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
    else:
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
    else:
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

def plot_SVM(exp_name, x_cols, weights, important_weight_num, remove_zero:bool=False):
    weight = abs(weight).sum(axis=0)
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
    else:
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



def combine(x_cols, feature_importance1:np.ndarray, feature_importance2:np.ndarray):
    top_x_cols1 = sorted(zip(x_cols, feature_importance1), key=lambda x: x[1], reverse=True)[:100]
    top_x_cols2 = sorted(zip(x_cols, feature_importance2), key=lambda x: x[1], reverse=True)[:100]

    top_x_cols1 = [x[0] for x in top_x_cols1]
    top_x_cols2 = [x[0] for x in top_x_cols2]

    # get intersection
    intersection = set(top_x_cols1).union(set(top_x_cols2))
    return [x for x in intersection]


def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion Matrix', cmap=plt.cm.Blues):
    """
    This function plots the confusion matrix.
    :param cm: Confusion matrix (NumPy array)
    :param classes: List of class labels
    :param normalize: Whether to normalize the matrix
    :param title: Title of the plot
    :param cmap: Colormap for the plot
    """
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def show_confusion_matrix(y_true, y_pred, block_output:bool):
    if not block_output:
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_true, y_pred)
        plot_confusion_matrix(cm, ['healthy', 'sick'])


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

def remove_data_leakage():
    Fields.add_field_to_unwanted_fields_file(4022,
        'leakage: Age pulmonary embolism (blood clot in lung) diagnosed')
    Fields.add_field_to_unwanted_fields_file(135,
        "leakage: Number of non-cancer illness")
    Fields.add_field_to_unwanted_fields_file(4012,
        "leakage: Age deep-vein thrombosis (DVT, blood clot in leg) diagnosed")
    Fields.add_field_to_unwanted_fields_file(5529,
        "leakage: Surgery on leg arteries = could be result of DVT")
    Fields.add_field_to_unwanted_fields_file(20009,
        "leakage: Interpolated Age of participant when non-cancer illness first diagnosed")
    Fields.add_field_to_unwanted_fields_file(20013,
        "leakage: Method of recording time when non-cancer illness first diagnosed")
    Fields.add_field_to_unwanted_fields_file(20008,
        "leakage: Interpolated Year when non-cancer illness first diagnosed")
    Fields.add_field_to_unwanted_fields_file(6150,
        "leakage: Vascular/heart problems diagnosed by doctor")
    Fields.add_field_to_unwanted_fields_file(87,
        "leakage: Non-cancer illness year/age first occurred")
    
    Encodings.add_values_to_unwanted_valuse_file(20002,'1094', 
        "leakage: None-cancer illness code, 1094 is DVT")
    Encodings.add_values_to_unwanted_valuse_file(20013,'-1', 
        "leakage: Method of recording time when non-cancer illness first diagnosed, related to 20002")
    Encodings.add_values_to_unwanted_valuse_file(41204,'Z867', 
        "leakage: Personal history of diseases of the circulatory system = removed")
    Encodings.add_values_to_unwanted_valuse_file(20002,'1093', 
        "leakage: 1093 is pulmonary embolism +/- dvt = removed")
    Encodings.add_values_to_unwanted_valuse_file(20013,'-4',
        "leakage: Method of recording time when non-cancer illness first diagnosed, related to 20002 = removed")
    message = "leakage: Personal history of diseases of the circulatory system"
    value = "Z867"
    Encodings.add_values_to_unwanted_valuse_file(41270,value,message)
    Encodings.add_values_to_unwanted_valuse_file(41204,value,message)
    Encodings.add_values_to_unwanted_valuse_file(41202,value,message)
    message = "leakage: I802 Phlebitis and thrombophlebitis of other deep vessels of lower extremities +- DVT"
    value = 'I802'
    Encodings.add_values_to_unwanted_valuse_file(41270,value,message)
    Encodings.add_values_to_unwanted_valuse_file(41204,value,message)
    Encodings.add_values_to_unwanted_valuse_file(41202,value,message)
    message = "leakage: I739 Peripheral vascular disease unspecified = could be DVT"
    value = 'I739'
    Encodings.add_values_to_unwanted_valuse_file(41270,value,message)
    Encodings.add_values_to_unwanted_valuse_file(41204,value,message)
    Encodings.add_values_to_unwanted_valuse_file(41202,value,message)
    message = "leakage: I801 Phlebitis and thrombophlebitis of femoral vein = not sure"
    value = 'I801'
    Encodings.add_values_to_unwanted_valuse_file(41270,value,message)
    Encodings.add_values_to_unwanted_valuse_file(41204,value,message)
    Encodings.add_values_to_unwanted_valuse_file(41202,value,message)
    message = "leakage: Z035 Observation for other suspected cardiovascular diseases"
    value = 'Z035'
    Encodings.add_values_to_unwanted_valuse_file(41270,value,message)
    Encodings.add_values_to_unwanted_valuse_file(41204,value,message)
    Encodings.add_values_to_unwanted_valuse_file(41202,value,message)
    message = "leakage: R609 Oedema/Edema can also develop as a result of a blood clot in the deep veins of the lower leg"
    value = 'R609'
    Encodings.add_values_to_unwanted_valuse_file(41270,value,message)
    Encodings.add_values_to_unwanted_valuse_file(41204,value,message)
    Encodings.add_values_to_unwanted_valuse_file(41202,value,message)
    message = "leakage: I803 Phlebitis and thrombophlebitis of lower extremities"
    value = 'I803'
    Encodings.add_values_to_unwanted_valuse_file(41270,value,message)
    Encodings.add_values_to_unwanted_valuse_file(41204,value,message)
    Encodings.add_values_to_unwanted_valuse_file(41202,value,message)
    messsage = 'leakage: Z921 Personal history of long-term (current) use of anticoagulants = drug to treat DVT'
    value = 'Z921'
    Encodings.add_values_to_unwanted_valuse_file(41270,value,messsage)
    Encodings.add_values_to_unwanted_valuse_file(41204,value,messsage)
    Encodings.add_values_to_unwanted_valuse_file(41202,value,messsage)
    message = 'leakage: I743 Embolism and thrombosis of arteries of the lower extremities'
    value = 'I743'
    Encodings.add_values_to_unwanted_valuse_file(41270,value,message)
    Encodings.add_values_to_unwanted_valuse_file(41204,value,message)
    Encodings.add_values_to_unwanted_valuse_file(41202,value,message)
    message = 'leakage: I800 Phlebitis and thrombophlebitis of superficial vessels of lower extremities'
    value = 'I800'
    Encodings.add_values_to_unwanted_valuse_file(41270,value,message)
    Encodings.add_values_to_unwanted_valuse_file(41204,value,message)
    Encodings.add_values_to_unwanted_valuse_file(41202,value,message)
    message = 'leakage: I269 Pulmonary embolism without mention of acute cor pulmonale'
    value = 'I269'
    Encodings.add_values_to_unwanted_valuse_file(41270,value,message)
    Encodings.add_values_to_unwanted_valuse_file(41204,value,message)
    Encodings.add_values_to_unwanted_valuse_file(41202,value,message)
    message = 'leakage: M7986 Other specified soft tissue disorders (Lower leg)'
    value = 'M7986'
    Encodings.add_values_to_unwanted_valuse_file(41270,value,message)
    Encodings.add_values_to_unwanted_valuse_file(41204,value,message)
    Encodings.add_values_to_unwanted_valuse_file(41202,value,message)
    

    
    
