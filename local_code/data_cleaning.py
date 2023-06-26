# this will include data cleaning specifics per category
import pandas as pd
import numpy as np

class cancer:
    relevant_collums = ["40021","40019","40009","40005","40008","40006","40013","40011","40012","41270"] 
    
    def data_cleaning(cancer_collums:pd.DataFrame):
        selected_cols = ["40009", # number of reports
                         "40008", # age in each report
                         "40011", # hystology
                         "40012"] # behavior
        cols = [
            col 
            for col in cancer_collums.columns 
            for x in selected_cols 
            if col in x
        ]
        relevant_data = cancer_collums[cols]
        return relevant_data[selected_cols]

class inpatient_summery:
    relevant_collums = ["41270","41280","41271","41281","41202","41262","41203","41263","41204","41205","41201"]

    def data_cleaning(inpatient_collums:pd.DataFrame):
        selected_cols = [
            # not relevant right now
        ]
        cols = [
            col 
            for col in inpatient_collums.columns 
            for x in selected_cols 
            if col in x
        ]
        relevant_data = inpatient_collums[cols]
        return relevant_data[selected_cols]
    
class circulatory_system_disorders:
    pass

class Assessment_centre_Medical_conditions:
    pass
    # extract prediction label here

class medication_and_supplement_use:
    pass


