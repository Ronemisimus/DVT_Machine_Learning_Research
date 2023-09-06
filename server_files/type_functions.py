import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.preprocessing import StandardScaler
from fields_and_encodings import Encodings

class TypeFunctions:
    
    def real(column:pd.DataFrame, values, field, train_size):
        # fill nulls with 0
        return column.fillna(0)


    def cat_one_hot(df:pd.DataFrame,values:np.ndarray, field, train_size):
        ######
        # turn a categorical to one hot
        ######

        ######
        # try to turn values to int if there is an exception then
        # try to turn data to int(cuz it's float and the values are in str(but in int format)) and after that to str if there is an exception 
        # turn data to str witout turning to int first (dataframe only turn values to float when they are only numerical)
        ######
        
        values = Encodings.remove_unwanted_values(field, values)

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
        if len(values)>0:
            mask = sum_of_col_appearance >= len(data)/len(values)*len(columns_names)*0.005
            data = data[:,mask]
            columns_names = columns_names[mask]
        res = pd.DataFrame(data=data,columns=columns_names)
        return res


    def Integer(column:pd.DataFrame, values, field, train_size):
        # fill nulls with 0
        return column.fillna(0).astype(np.int64)


type_func_dict = {
    "Integer": TypeFunctions.Integer,
    "Categorical (single)": TypeFunctions.cat_one_hot,
    "Categorical (multiple)": TypeFunctions.cat_one_hot,
    "Continuous": TypeFunctions.real
}


class ColumnCleaner:
    def __init__(self, column_names, field_type, field_values, field_name):
        self.column_names = column_names
        self.field_type = field_type
        self.field_values = field_values
        if self.field_type not in ['Categorical (single)', 'Categorical (multiple)'] and \
            len(self.column_names)>0:
            self.scaler = StandardScaler()
        else:
            self.scaler = None
        self.field_name = field_name
    
    def fit_transform(self, df:pd.DataFrame, train_size):
        df = type_func_dict[self.field_type](df, self.field_values, str(self.field_name), len(df))
        if self.scaler:
            names = df.columns
            df_train = self.scaler.fit_transform(df.head(train_size))
            df_test = self.scaler.transform(df.tail(len(df)-train_size))
            df = np.concatenate([df_train, df_test])
            df = pd.DataFrame(df,columns=names)
        return df
    
    def transform(self, df:pd.DataFrame, train_size):
        df = type_func_dict[self.field_type](df, self.field_values, str(self.field_name), 0)
        if self.scaler:
            names = df.columns
            df_test = self.scaler.transform(df.tail(len(df)-train_size))
            df_train = df.head(train_size).to_numpy()
            df = np.concatenate([df_train, df_test])
            df = pd.DataFrame(df,columns=names)
        return df