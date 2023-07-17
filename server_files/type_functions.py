import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer

class TypeFunctions:
    
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

    def Integer(s, values, field):
        # for now an integer is fine
        return s.fillna(0).astype(np.int64)