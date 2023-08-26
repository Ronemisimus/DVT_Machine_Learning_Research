import pandas as pd
import numpy as np
import ast


class Fields:
 
    def load_desired_fields():
        fields = pd.read_csv("csv/field_list.csv")
        field_names = fields['field_id'].to_numpy().astype(np.int64).flatten()
        field_types = fields['value_type'].to_numpy().astype(str).flatten()
        field_encodings = fields['encoding_id'].to_numpy().astype(str).flatten()

        # remove undesired fields
        field_names, field_types, field_encodings = Fields.remove_unwanted_fields(field_names, field_types, field_encodings)

        del fields
        return field_names, field_types, field_encodings


    def remove_unwanted_fields(field_names, field_types, field_encodings):
        # read unwanted_field_list.csv file and remove the unwanted fields
        # returns clean data 
        try:
            unwanted_fields = pd.read_csv("csv/unwanted_field_list.csv")
        except:
            return field_names, field_types, field_encodings
        
        unwanted_fields_name = unwanted_fields['field_id'].to_numpy().astype(np.int64).flatten()

        for field in field_names :
            if field in unwanted_fields_name:
                index = np.where(field_names==field)[0][0]
                field_names = np.delete(field_names, index)
                field_types = np.delete(field_types, index)
                field_encodings = np.delete(field_encodings, index)

        return field_names, field_types, field_encodings
    
    def add_field_to_unwanted_fields_file(field_name, reason):
        # add to unwanted_field_list.csv the field_name
        # if unwanted_field_list.csv doesn't exists then create file then add

        field_name = str(field_name)

        try:
            unwanted_fields = pd.read_csv("csv/unwanted_field_list.csv")
            unwanted_fields_name = unwanted_fields['field_id'].to_numpy().astype(str).flatten()
            if str(field_name) in unwanted_fields_name:
                return
        except:
            unwanted_fields = pd.DataFrame(columns=['field_id', 'reason'])

        new_field = {'field_id': field_name,'reason': reason}
        unwanted_fields.loc[len(unwanted_fields) + 1] = new_field
        unwanted_fields.to_csv("csv/unwanted_field_list.csv", index=False)


class Encodings:

    def create_category_dictionary():
        # read the encoding_dict.csv 
        # and return dictionary
        # key (encoding_id) value(dictionary of encoding possible values)

        dictionary = {}
        encodings = pd.read_csv("csv/encoding_dict.csv",sep="\t")
        for i in range(len(encodings.index)):
            str_of_encoding_values = '[' + encodings.values[i][1][1:-1] + ']'
            list_of_encoding_values = ast.literal_eval(str_of_encoding_values)
            dictionary[str(encodings.values[i][0])] = np.array(list_of_encoding_values,dtype=str)

        return dictionary


    def add_values_to_unwanted_valuse_file(field, value, reason):
        # add to unwanted_values.csv the field and value
        # if unwanted_values.csv doesn't exists then create file then add

        value = str(value)
        field = str(field)

        try:
            unwanted_values = pd.read_csv("csv/unwanted_values.csv")
        except:
            unwanted_values = pd.DataFrame(columns=["fields","value","reason"])
            unwanted_values.loc[len(unwanted_values)+1] = [field, {value},reason]
            unwanted_values.to_csv("csv/unwanted_values.csv", index=False)
            return
        
        unwanted_values_fields = unwanted_values['fields'].to_numpy().astype(str).flatten()
        index = np.where(unwanted_values_fields==field)[0]
        
        if np.size(index):
            unwanted_values_value = unwanted_values['value'].to_numpy().astype(str).flatten()
            list_of_encoding_values = ast.literal_eval(unwanted_values_value[index[0]])
            if value in list_of_encoding_values:
                return
            else:
                list_of_encoding_values.add(value)
                unwanted_values.at[index[0],'value'] = list_of_encoding_values
                unwanted_values.at[index[0],'reason'] = unwanted_values.at[index[0],'reason'] + ', ' + reason
                unwanted_values.to_csv("csv/unwanted_values.csv", index=False)
        else:
            unwanted_values.loc[len(unwanted_values)+1] = [field, {value},reason]
            unwanted_values.to_csv("csv/unwanted_values.csv", index=False)
    

    def remove_unwanted_values(field, values):

        try:
            unwanted_values = pd.read_csv("csv/unwanted_values.csv")
        except:
            return values
        
        unwanted_values_fields = unwanted_values['fields'].to_numpy().astype(str).flatten()
        unwanted_values_value = unwanted_values['value'].to_numpy().astype(str).flatten()
        
        index = np.where(unwanted_values_fields==field)[0]

        if np.size(index):
            list_of_unwanted_encoding_values = ast.literal_eval(unwanted_values_value[index[0]])
            values = [val for val in values if not val in list_of_unwanted_encoding_values]
        
        return np.array(values,dtype=str)
