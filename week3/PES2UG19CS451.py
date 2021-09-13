'''
Assume df is a pandas dataframe object of the dataset given
'''



import numpy as np
import pandas as pd
import random




'''Calculate the entropy of the enitre dataset'''
def get_entropy_of_dataset(df):
    entropy = 0

    if df.empty:
        return entropy
    
    lastColumnName = df.columns[-1]
    targetVal = df[lastColumnName].unique()
    
    for tarVal in targetVal:
        p = df[lastColumnName].value_counts()[tarVal] / len(df[lastColumnName])
        if(p == 0):
            continue
        entropy += -(p * np.log2(p))

    return entropy




'''Return entropy of the attribute provided as parameter'''
def get_avg_info_of_attribute(df,attribute):    
    entropyOfAttributes = 0
    
    if df.empty:
        return entropyOfAttributes

    try:
        lastColumnName = df.columns[-1]
        targetVal = df[lastColumnName].unique()
        attributeValues = df[attribute].unique()

        for attVal in attributeValues:
            entropy_of_features = 0
            p_den = len(df[attribute][df[attribute] == attVal])
            
            for tarVal in targetVal:
                p_num = len(df[attribute][df[attribute] == attVal][df[lastColumnName] == tarVal])
                p = p_num / p_den
    
                if(p == 0):
                    continue
                entropy_of_features += -(p * np.log2(p))
            
            entropyOfAttributes += (((p_den/len(df)) * entropy_of_features))

    except KeyError:
        print("Attribute:", attribute, "not in dataset.")

    return entropyOfAttributes




'''Return Information Gain of the attribute provided as parameter'''
def get_information_gain(df,attribute):
    information_gain = 0
    try:
        information_gain = get_entropy_of_dataset(df) - get_avg_info_of_attribute(df,attribute)

    except KeyError:
        print("Attribute:", attribute, "not in dataset.")

    return information_gain




''' Return Attribute with highest info gain'''
def get_selected_attribute(df):
    if df.empty or len(df.columns) == 1:
        return (dict(),'')

    attributes = list(df)[:-1]
    attribute_information_gain = list(map(lambda x:get_information_gain(df,x),attributes))
    information_gains = dict(zip(attributes,attribute_information_gain))
    max_info_gain_index, _ = max(enumerate(attribute_information_gain),key = lambda x:x[1])
    selected_column = attributes[max_info_gain_index]

    '''
    Return a tuple with the first element as a dictionary which has IG of all columns
    and the second element as a string with the name of the column selected

    example : ({'A':0.123,'B':0.768,'C':1.23} , 'C')    '''
    return (information_gains,selected_column)
    pass
