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
from matplotlib import pyplot as plt

# hyper params
# hyper params


def real(column, values, field):
    # usually we wan't to keep it the same 
    # but might want to normallize it 
    return column.fillna(0)


def cat_one_hot(df:pd.DataFrame,values:np.ndarray, field):
    ######
    # turn a categorical to one hot
    ######


    ######
    # try to turn values to int if there is an exception then
    # try to turn data to int(cuz it's float and the values are in str(but in int format)) and after that to str if there is an exception 
    # turn data to str witout turning to int first (dataframe only turn values to float when they are only numerical)
    ######
    try:
        values = values.astype(int)
        enc = MultiLabelBinarizer(classes=values)
        data = df.to_numpy()
        data = enc.fit_transform(data.astype(np.float32))
    except ValueError:
        # 1.0 != 1
        df = df.fillna("<empty>").astype('str')
        enc = MultiLabelBinarizer(classes=values)
        data = df.to_numpy()
        data = enc.fit_transform(data)

    columns_names = np.array([field + '.' + str(val) for val in values])
    sum_of_col_appearance = np.sum(data, axis=0)

    ######
    # the number here could be adjusted (0.005)
    ######
    mask = sum_of_col_appearance >= len(data)/len(values)*len(columns_names)*0.005

    data = data[:,mask]
    columns_names = columns_names[mask]
    res = pd.DataFrame(data=data,columns=columns_names)
    return res


def create_category_dictionary():
    # read the encoding_dict.cvs 
    # and return dictionary
    # key (encoding_id) value(dictionary of encoding possible values)

    dict = {}
    encodings = pd.read_csv("encoding_dict.csv",sep="\t")
    for i in range(len(encodings.index)):
        sub_dict = {}
        str_of_encoding_values = '[' + encodings.values[i][1][1:-1] + ']'
        list_of_encoding_values = ast.literal_eval(str_of_encoding_values)
        dict[str(encodings.values[i][0])] = np.array(list_of_encoding_values,dtype=str)

    return dict

def Integer(s, values, field):
    # for now an integer is fine
    return s.fillna(0).astype(np.int64)

type_func_dict = {
    "Integer":Integer,
    "Categorical (single)":cat_one_hot,
    "Categorical (multiple)":cat_one_hot,
    "Continuous":real
}

def load_desired_fields():
    fields = pd.read_csv("field_list.csv")
    field_names = fields['field_id'].to_numpy().astype(np.int64).flatten()
    field_types = fields['value_type'].to_numpy().astype(str).flatten()
    field_encodings = fields['encoding_id'].to_numpy().astype(str).flatten()
    del fields
    return field_names, field_types, field_encodings

def final_columns(field_names):
    positive = pd.read_csv('instance_0_positive.csv',nrows=1)
    selected_cols = [
        col 
        for field in field_names 
        for col in positive.columns 
        if col.startswith(str(field) + "-0")
        # we only wan't exacly the field asked from instance 0
    ]
    return selected_cols

def create_mixed_dataset(selected_cols):
    positive = pd.read_csv('instance_0_positive.csv',usecols=selected_cols,chunksize=1000,low_memory=False)
    negative = pd.read_csv('all_instance_negative.csv',usecols=selected_cols,chunksize=1000,low_memory=False)

    cols = positive.get_chunk(0).columns
    print(len(cols))
    s_chunk = pd.DataFrame(columns=cols)
    s_chunk.to_csv('shuffled_dataset_first_itter.csv',index=False,mode='w')
    for p_chunk, n_chunk in tqdm.tqdm(zip(positive,negative), total=11):
        s_chunk = pd.concat([p_chunk,n_chunk]).sample(frac=1)
        s_chunk[cols].to_csv('shuffled_dataset_first_itter.csv',mode='a',header=False,index=False)
    del positive,negative, p_chunk, s_chunk, n_chunk


def prepare_data(exp_name):
    # get desired field list
    field_names, field_types, field_encodings = load_desired_fields()

    # build final field list including instances and arrays...
    selected_cols = final_columns(field_names)

    # build final dataset from positive and negative with only needed columns
    create_mixed_dataset(selected_cols)

    # create a clean dataset with clean datatypes
    dataset = pd.read_csv('shuffled_dataset_first_itter.csv', chunksize=20130, low_memory=False)
    cols = dataset.get_chunk(0).columns

    #####
    # these lines were removed here, added after cleaning the data
    # s_chunk = pd.DataFrame(columns=cols)
    # s_chunk.to_csv('shuffled_dataset_clean.csv',index=False,mode='w')
    #####

    # build final field to type dictionary
    field_groups = {}
    for f, t, e in tqdm.tqdm(zip(field_names,field_types,field_encodings), total=len(field_names)):
        # we wan't to avoid changing the final label
        if f != 6152:
            res = []
            f = str(f)
            for col in cols:
                if col.startswith(f + '-'):
                    res.append(col)
            field_groups[f] = (res,type_func_dict[t], e)

    print("done with type matching")

    # build dict of encoding values
    enc_dict = create_category_dictionary()

    # copy chunks while cleaning dtypes
    for i,chunk in enumerate(dataset):
        res_cols = []
        for i, (field, (res,clean_func, encoding_id)) in enumerate(tqdm.tqdm(field_groups.items(),desc=f"chunk {i+1}/1")):
            clean_data = clean_func(chunk[res], enc_dict.get(encoding_id, []), str(field))
            chunk.drop(columns=res, inplace=True)
            if not clean_data.empty:
                res_cols.append(clean_data)

        
        ######
        # Added the 6152 here 
        # there should be a better way to add the 6152 but fuck it
        res = []
        for col in cols:
                if col.startswith('6152-'):
                    res.append(col)
        res_cols.append(chunk[res])
        ######

        chunk = pd.concat(res_cols,axis=1)

        ######
        # Added the columns here since the old way we added the coulmns without the name change 
        # i.e 41202-0.0 if categorical became 41202-.Z450
        # and the old way would have 41202-0.0 as a col and not 41202-.Z450
        # again there should be a better way to add this but fuck it
        cols = chunk.columns
        s_chunk = pd.DataFrame(columns=cols)
        s_chunk.to_csv('shuffled_dataset_clean.csv',index=False,mode='w')
        ######

        chunk.to_csv('shuffled_dataset_clean.csv',mode='a',header=False,index=False)
        logging.info("shuffled_dataset_clean.csv is created")

    # if all works this will print only number and float dtypes
    print(np.unique(chunk.dtypes))

def load_data(exp_name):
    # seperate x and y fields
    dataset = pd.read_csv('shuffled_dataset_clean.csv', nrows=1)

    # 6152 is the field we want to predict
    y_cols = [col for col in dataset.columns if col.startswith("6152-")]
    x_cols = [col for col in dataset.columns if not col.startswith("6152-")]
    
    dataset = pd.read_csv('shuffled_dataset_clean.csv', usecols=x_cols)
    X = dataset.to_numpy()
    dataset = pd.read_csv('shuffled_dataset_clean.csv', usecols=y_cols)
    Y = np.sum(dataset[y_cols].to_numpy()==5,axis=-1)!=0
    
    return X,Y, x_cols

def experiment(exp_name,remake_dataset):
    #####
    # basically just added a logging system to keep my sanity
    # add a folder named logs otherwise this will crash
    #####

    logging.basicConfig(filename='logs/'+ exp_name + "__" + str(datetime.datetime.now())
                        +".log", encoding = 'utf-8', level=logging.DEBUG)
    logging.info("start: " + str(datetime.datetime.now()))

    if remake_dataset:
        logging.info("remake dataset")
        prepare_data(exp_name)
    X,Y, x_cols = load_data(exp_name)

    logging.info("loaded data")
    clf = LogisticRegression(penalty='l1',solver='liblinear',max_iter=200,verbose=True)
    train_limit = X.shape[0]*7//10
    logging.info("before fit")
    clf.fit(X[:train_limit],Y[:train_limit])
    logging.info("after fit")

    test_score = clf.score(X[train_limit:],Y[train_limit:])
    train_score = clf.score(X[:train_limit],Y[:train_limit])
    logging.info("test score :" + str(test_score))
    logging.info("train score :" + str(train_score))
    logging.info("end: " + str(datetime.datetime.now()))
    print(train_score, test_score)

    s = pickle.dumps(clf)
    with open('logs/'+ exp_name+".pkl",'wb') as f_out:
        f_out.write(s)

    return clf, x_cols

def plot_fields(exp_name, x_cols, weights, important_weight_num):
    data = sorted(zip(x_cols, weights),key=lambda x: x[1],reverse=True)
    sorted_x_cols = [x[0] for x in data]
    weights = [x[1] for x in data]
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
