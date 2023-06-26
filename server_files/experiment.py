import pandas as pd
import numpy as np
import tqdm
import pickle

def real(column):
    # usually we wan't to keep it the same 
    # but might want to normallize it 
    return column.fillna(0)

def cat_single(df:pd.DataFrame):
    # this makes categorial data become numerical
    # codes of each category
    df = df.fillna("<empty>").astype('str')
    all_categories = sorted(np.unique(df.to_numpy()))

    for col in df.columns:
        df[col] = pd.Categorical(
            df[col], 
            categories=all_categories
        ).codes
    return df

def cat_multiple(df:pd.DataFrame):
    # for now we'll do the same
    df = df.fillna("<empty>").astype('str')
    all_categories = sorted(np.unique(df.to_numpy()))

    for col in df.columns:
        df[col] = pd.Categorical(
            df[col], 
            categories=all_categories
        ).codes
    return df

def Integer(s):
    # for now an integer is fine
    return s.fillna(0).astype(np.int64)

type_func_dict = {
    "Integer":Integer,
    "Categorical (single)":cat_single,
    "Categorical (multiple)":cat_multiple,
    "Continuous":real
}

def load_desired_fields():
    fields = pd.read_csv("field_list.csv")
    field_names = fields['field_id'].to_numpy().astype(np.int64).flatten()
    field_types = fields['value_type'].to_numpy().astype(str).flatten()
    del fields
    return field_names, field_types

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
    field_names, field_types = load_desired_fields()

    # build final field list including instances and arrays...
    selected_cols = final_columns(field_names)

    # build final dataset from positive and negative with only needed columns
    create_mixed_dataset(selected_cols)

    # create a clean dataset with clean datatypes
    dataset = pd.read_csv('shuffled_dataset_first_itter.csv', chunksize=20130, low_memory=False)
    cols = dataset.get_chunk(0).columns
    s_chunk = pd.DataFrame(columns=cols)
    s_chunk.to_csv('shuffled_dataset_clean.csv',index=False,mode='w')

    # build final field to type dictionary
    field_groups = {}
    for f, t in tqdm.tqdm(zip(field_names,field_types), total=len(field_names)):
        # we wan't to avoid changing the final label
        if f != 6152:
            res = []
            f = str(f)+"-"
            for col in cols:
                if col.startswith(f):
                    res.append(col)
            field_groups[f] = (res,type_func_dict[t])

    print("done with type matching")

    # copy chunks while cleaning dtypes
    for i,chunk in enumerate(dataset):
        for field, (res,clean_func) in tqdm.tqdm(field_groups.items(),desc=f"chunk {i+1}/1"):
            if clean_func in [cat_single, cat_multiple]:
                for col in res:
                    if chunk[col].dtype == int:
                        chunk[col] = chunk[col].astype(float)
            chunk[res] = clean_func(chunk[res])
        chunk[cols].to_csv('shuffled_dataset_clean.csv',mode='a',header=False,index=False)

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
    if remake_dataset:
        prepare_data(exp_name)
    X,Y, x_cols = load_data(exp_name)
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import normalize
    clf = LogisticRegression(penalty='l1',solver='liblinear',max_iter=200,verbose=True)
    train_limit = X.shape[0]*7//10
    clf.fit(X[:train_limit],Y[:train_limit])
    test_score = clf.score(X[train_limit:],Y[train_limit:])
    train_score = clf.score(X[:train_limit],Y[:train_limit])
    print(train_score, test_score)
    s = pickle.dumps(clf)
    with open(exp_name+".pkl",'wb') as f_out:
        f_out.write(s)

    return clf, x_cols

def plot_fields(x_cols, weights, important_weight_num):
    from matplotlib import pyplot as plt
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
    plt.show()
