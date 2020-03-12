import pandas as pd
import numpy as np

month_dictionary = {
    'January': 1,
    'February': 2, 
    'March': 3,
    'April': 4, 
    'May': 5, 
    'June': 6, 
    'July': 7, 
    'August': 8, 
    'September': 9, 
    'October': 10, 
    'November': 11, 
    'December': 12}

def apply_categorical_datacleaning(data_input, agent_threshold = 500, company_threshold = 100):
    """
    This method takes the dataframes, in H1.csv or H2.csv's format, 
    and applies the necessary data cleaning steps on the 13 categorical
    variables. Reasoning is included in the 01_datacleaning.ipyb file. 
    
    Parameters
    ----------
    data_input: pandas DataFrame to modify
    agent_threshold: the rows where value in 'Agent' column is below this will be 
        recoded as 'Other'
    company_threshold: the rows where value in 'Company' column is below this will be 
        recoded as 'Other'
            
    Returns
    ----------
    data: pandas DataFrame after the data cleaning steps
    
    """
    
    # keeping the original data unchanged
    data = data_input.copy()
    
    # Step 1: recode ArrivalDateMonth as number
    data['ArrivalDateMonth'] = data['ArrivalDateMonth'].map(lambda x: month_dictionary[x])
    data['ArrivalDateMonth'].astype('int64')
    
    # Step 2: recode Country to three categories
    data['Country'].fillna('Unknown', inplace = True)
    data['Country'] = data['Country'].map(lambda x: recode_country(x))
    
    # Step 3: drop 'Undefined' row from DistributionChannel
    data.drop(data[data['DistributionChannel'] == 'Undefined'].index, axis = 0, inplace = True)
    
    # Step 4: drop 'P' and 'L' from ReservedRoomType and AssignedRoomType
    data.drop(data[data['ReservedRoomType'] == 'P               '].index, axis = 0, inplace = True)
    data.drop(data[data['ReservedRoomType'] == 'L               '].index, axis = 0, inplace = True)
    data.drop(data[data['AssignedRoomType'] == 'P               '].index, axis = 0, inplace = True)
    data.drop(data[data['AssignedRoomType'] == 'L               '].index, axis = 0, inplace = True)
    
    # Step 5: create new boolean, ReservedTypeEqualsAssigned, 1 if two are equal, 0 if not
    data['ReservedTypeEqualsAssigned'] = (data['ReservedRoomType'] == data['AssignedRoomType']) * 1
    
    # Step 6: recode rows where agent is from a group with less than agent_threshold frequency
    agent_freq_dictionary = data['Agent'].value_counts().to_dict()
    data['Agent'] = data['Agent'].map(lambda x: recode_by_threshold(x, agent_freq_dictionary, agent_threshold))
    
    # Step 7: recode rows where company is from a group with less than company_threshold frequency
    company_freq_dictionary = data['Company'].value_counts().to_dict()
    data['Company'] = data['Company'].map(lambda x: recode_by_threshold(x, company_freq_dictionary, company_threshold))
    
    # Step 8: drop ReservationStatus and ReservationStatusDate from the database
    columns_to_drop = ['ReservationStatus', 'ReservationStatusDate']
    data.drop(columns_to_drop, axis = 1, inplace = True)

    # Final step: returns the data object
    return data

def recode_country(country):
    """
    Recodes the country categories to a manageable number of groups.
    
    Parameters
    ----------
    country: a three-character country code
    
    Returns
    ----------
    country_group: Portugal, NonPortugal, Unknown
    
    """
    
    if country == 'PRT':
        country_group = 'Portugal'
    elif country == 'Unknown':
        country_group = 'Unknown'
    else:
        country_group = 'NonPortugal'
        
    return country_group

def recode_by_threshold(id_number, freq_dictionary, threshold):
    """
    Groups the rows with less frequency into an 'Other' group. 
    Also renames 'NULL' to 'Unknown', and strips the strings
    
    Parameters
    ----------
    agent_number: ID number of currently checked agent
    agent_dictionary: value_counts() dictionary of 'Agent' column
    agent_threshold: agents below this threshold are put in an 'Other' category
    
    Returns
    ----------
    agent_group: grouped agent
    """
    
    if id_number.strip() == 'NULL':
        group = 'Unknown'
    else:
        frequency = freq_dictionary[id_number]
        if frequency < threshold:
            group = 'Other'
        else:
            group = id_number.strip()
            
    return group


def apply_continuous_datacleaning(data_input):
    """
    This method takes the dataframe, in H1.csv or H2.csv's format, 
    and applies the necessary data cleaning steps on the continuous
    variables. Reasoning is included in the 01_datacleaning.ipyb file. 
    
    Parameters
    ----------
    data_input: pandas DataFrame to modify
            
    Returns
    ----------
    data: pandas DataFrame after the data cleaning steps
    deleted_rownum: a dictionary where keys are the column names, 
        values are the number of rows deleted due to constraints 
        of that specific column
    
    """
    
    # keeping the original data unchanged
    data = data_input.copy()
    deleted_rownum = {}
    last_deleted_rownum = 0
    current_rownum = len(data)
    
    # Step 1: delete rows where Adults > 4
    data.drop(data[data['Adults'] >4].index, axis = 0, inplace = True)
    last_deleted_rownum = current_rownum - len(data)
    deleted_rownum['Adults'] = last_deleted_rownum
    current_rownum = len(data)
    
    # Step 2: delete rows where Children > 4
    data.drop(data[data['Children'] >4].index, axis = 0, inplace = True)
    last_deleted_rownum = current_rownum - len(data)
    deleted_rownum['Children'] = last_deleted_rownum
    current_rownum = len(data)
    
    # Step 3 drop rows where PreviousCancellations > 1, then drop the whole column
    data.drop(data[data['PreviousCancellations'] >1].index, axis = 0, inplace = True)
    data.drop(['PreviousCancellations'], axis = 1, inplace = True)
    last_deleted_rownum = current_rownum - len(data)
    deleted_rownum['PreviousCancellations'] = last_deleted_rownum
    current_rownum = len(data)
    
    # Step 4: replace RequiredCarParkingSpaces to max 2
    data['RequiredCarParkingSpaces'] = data['RequiredCarParkingSpaces'].map(lambda x: np.minimum(x, 2))
    
    # Step 5: create total stay nights, drop rows where it is 0 (they are also columns where ADR = 0)
    data['StaysInNights'] = data['StaysInWeekendNights'] + data['StaysInWeekNights']
    data.drop(data[data['StaysInNights'] == 0].index, axis = 0, inplace = True)
    last_deleted_rownum = current_rownum - len(data)
    deleted_rownum['StaysInNights'] = last_deleted_rownum
    current_rownum = len(data)
    
    return data, deleted_rownum
