import pandas as pd
import numpy as np


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
    print(len(cat_fields))
    res = pd.concat([
        cat_fields["field_id"],
        cat_fields["value_type"].replace(value_type_reverse).astype(pd.StringDtype())
    ], axis=1)
    print(len(res))

    res.to_csv("final_field_list.csv", index=False)


if __name__ == '__main__':
    main()