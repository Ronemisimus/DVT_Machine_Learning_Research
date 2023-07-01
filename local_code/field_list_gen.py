import pandas as pd
import numpy as np
import tqdm

encoding_values_files = ["schema/ehierint.txt", 
                             "schema/ehierstring.txt", 
                             "schema/esimpdate.txt",
                             "schema/esimpint.txt", 
                             "schema/esimpreal.txt",
                             "schema/esimpstring.txt"]

def check_encoding_avaliable(encoding_codes):
    # test all codes are in the files
    
    encs = []
    for f in encoding_values_files:
        f_enc = pd.read_csv(f, sep="\t", usecols=["encoding_id"],encoding = "ISO-8859-1")
        f_enc = np.unique(f_enc.values)
        encs = np.unique(encs + f_enc.tolist()).tolist()

    encoding_codes = np.unique(encoding_codes.values).tolist()

    # 0 means it's a real number with no encoding
    encoding_codes.remove(0)

    missing_encodings =[value for value in encoding_codes if value not in encs]
        

    if len(missing_encodings) == 0:
        return encoding_codes
    else:
        print("misssing encoding for a field. missing encodings are: \n", missing_encodings)
        print("------------------------------------------------")
        print("missing encodings count is:",len(missing_encodings))
        exit()

def process_dict(encoding_codes):
    encodings  = check_encoding_avaliable(encoding_codes)
    enc_dicts = {}
    for enc in tqdm.tqdm(encodings):
        vals = set()
        for f in encoding_values_files:
            if f in ["schema/ehierint.txt", "schema/ehierstring.txt"]:
                f_enc = pd.read_csv(f, sep="\t", usecols=["encoding_id", "value", "selectable"],encoding = "ISO-8859-1")
            else:    
                f_enc = pd.read_csv(f, sep="\t", usecols=["encoding_id", "value"],encoding = "ISO-8859-1")
            f_enc = f_enc[f_enc["encoding_id"]==enc]
            if f in ["schema/ehierint.txt", "schema/ehierstring.txt"]:
                f_enc = f_enc[f_enc["selectable"]==1]
            vals = vals.union(f_enc["value"].tolist())
        enc_dicts[enc] = vals
    enc_dicts = pd.DataFrame(enc_dicts.items(), columns=['encoding_id','values'])
    enc_dicts.to_csv("encoding_dict.csv", index=False,sep='\t')


def main():
    cat_list = pd.read_csv("desired_cat_list.csv",)
    cat_list = cat_list.to_numpy().astype(int).flatten()

    field_scheme = pd.read_csv("schema/field.txt",sep="\t")
    print(len(field_scheme))
    cat_fields = field_scheme[field_scheme["main_category"].isin(cat_list)]
    print(len(cat_fields))
    value_type = {    
        "Integer": 11,
        "Categorical (single)": 21,
        "Categorical (multiple)": 22,
        "Continuous": 31,
        "Text": 41,
        "Date": 51,
        "Time": 61,
        "Compound": 101
    }

    value_type_reverse = {v:k for k,v in value_type.items()}

    desired_value_types = [11,21,22,31]
    cat_fields = cat_fields[cat_fields['value_type'].isin(desired_value_types)]

    # place to filter encodings
    
    # place to filter encodings

    process_dict(cat_fields["encoding_id"])

    print(len(cat_fields))
    res = pd.concat([
        cat_fields["field_id"],
        cat_fields["value_type"].replace(value_type_reverse).astype(pd.StringDtype()),
        cat_fields["encoding_id"]
    ], axis=1)
    print(len(res))

    res.to_csv("final_field_list.csv", index=False)
    


if __name__ == '__main__':
    main()