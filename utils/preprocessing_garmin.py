import pandas as pd
import numpy as np
import sklearn
import copy
from os.path import exists

def read_subject(id):
    """read epochs, heart rates and sleep data for one subject

    Parameters
    ----------
    id: str
        string indicating the id of a subject

    Returns
    -------
    epochs: pd.DataFrame
        pandas dataframe detailing Garmin data for each epoch (15 minute intervals)

    hrates: pd.DataFrame
        pandas dataframe detailing heart rate data recorded by the Garmin device
    
    sleeps: pd.DataFrame
        pandas dataframe detailing sleep period records determined by the Garmin device
    """
    epochs_fname = "./data/epochs_cleaned/epochs_cleaned_{}.csv".format(
        id)
    hrates_fname = "./data/hr_cleaned/hr_cleaned_{}.csv".format(
        id)
    sleeps_fname = "./data/sleep_cleaned/sleep_cleaned_{}.csv".format(
        id)

    epochs = pd.read_csv(epochs_fname)
    hrates = pd.read_csv(hrates_fname)
    sleeps = pd.read_csv(sleeps_fname)

    return epochs, hrates, sleeps


def aggregate_by_subject(studyId):
    """produce aggregated dataframe from epochs, hrates and sleeps

    Parameters
    ----------
    studyId: str
        string indicating the id of a subject

    Returns
    -------
    data_table: pd.DataFrame
        pandas dataframe of the aggregated garmin data, marking every epoch with sleep labels
    """
    # if the aggregated dataframe already exists for that subject, skip this process
    aggregate_filepath = "./data/aggregate/aggregate_{}.csv".format(studyId)
    if exists(aggregate_filepath):
        return None

    epochs, hrates, sleeps = read_subject(studyId)

    minTimeStamp = min(epochs["startTimeStamp"])
    maxTimeStamp = max(epochs["startTimeStamp"])

    firstTimeStamp = int(minTimeStamp - minTimeStamp % 900)
    lastTimeStamp = int(maxTimeStamp - maxTimeStamp % 900)

    startTimeStamps = range(firstTimeStamp, lastTimeStamp+900, 900)
    endTimeStamps = range(firstTimeStamp+900, lastTimeStamp+1800, 900)

    nrows = len(startTimeStamps)
    col_names = ["studyId",
                 "startTimeStamp",
                 "startDateTime",
                 "endTimeStamp",
                 "steps",
                 "distanceInMeters",
                 "maxMotionIntensity",
                 "meanMotionIntensity",
                 "sleepLabel"
                 ]

    col_names = ["{0:03d}_heartRate".format(
        i*15) for i in range(60)] + col_names
    data_table = pd.DataFrame(columns=col_names)
    data_table["startTimeStamp"] = startTimeStamps
    data_table["endTimeStamp"] = endTimeStamps
    data_table["studyId"] = [studyId for i in range(nrows)]

    n_epochs = epochs.shape[0]
    # fill in epochs first
    epochs.head()
    d_index = 0
    for i in range(n_epochs):
        current_timeStamp = epochs.loc[i, "startTimeStamp"]
        d_index_old = d_index
        while True:
            if d_index >= data_table.shape[0]:
                # print("Could not find that epochs")
                d_index = d_index_old
                break
            elif (current_timeStamp >= data_table.loc[d_index, "startTimeStamp"]) and \
                    (current_timeStamp < data_table.loc[d_index, "startTimeStamp"] + 900):
                data_table.loc[d_index, "steps"] = epochs.loc[i, "steps"]
                data_table.loc[d_index,
                               "distanceInMeters"] = epochs.loc[i, "distanceInMeters"]
                data_table.loc[d_index,
                               "maxMotionIntensity"] = epochs.loc[i, "maxMotionIntensity"]
                data_table.loc[d_index, "meanMotionIntensity"] = epochs.loc[i,
                                                                            "meanMotionIntensity"]
                data_table.loc[d_index, "startDateTime"] = epochs.loc[i,
                                                                      "startDate"] + " " + epochs.loc[i, "startTime"]
                break
            else:
                d_index += 1
    print("Finished Loading Epochs")
    ##### put in heart rate
    hr_index = 0
    for i in range(nrows):
        base_timeStamp = data_table["startTimeStamp"][i]
        hr_index_old = hr_index
        for k in range(60):
            current_timeStamp = base_timeStamp + k*15
            while True:
                if hr_index >= hrates.shape[0]:
                    # print("heart rate record not found")
                    hr_index = hr_index_old
                    break
                elif current_timeStamp > hrates["timeStamp"][hr_index]:
                    hr_index += 1
                elif current_timeStamp == hrates["timeStamp"][hr_index]:
                    data_table.iloc[i, k] = hrates["heartRate"][hr_index]
                    break
                else:
                    break
    print("Finished Loading Heart Rates")
    ##### put in sleep label
    data_table["sleepLabel"] = [0 for i in range(nrows)]
    n_sleeps = sleeps.shape[0]
    d_index = 0
    for i in range(n_sleeps):
        sleepTimeStamp = sleeps["startTimeStamp"][i]
        endTimeStamp = sleeps["endTimeStamp"][i]
        d_index_old = d_index
        while True:
            if data_table["endTimeStamp"][d_index] > endTimeStamp:
                # print("sleep record not found")
                d_index = d_index_old
                break
            elif data_table["startTimeStamp"][d_index] < sleepTimeStamp:
                d_index += 1
            elif data_table["startTimeStamp"][d_index] >= sleepTimeStamp and data_table["endTimeStamp"][d_index] <= endTimeStamp:
                data_table.loc[d_index, "sleepLabel"] = 1
                d_index += 1
    print("Finished Loading Sleeps")
    data_table.to_csv("./data/aggregate/aggregate_{}.csv".format(studyId))
    return data_table


def read_aggregate(studyId):
    """read the aggregate dataframe for a subject

    Parameters
    ----------
    studyId: str
        string indicating the id of a subject

    Returns
    -------
    df_id: pd.DataFrame
        pandas dataframe of the aggregated garmin data, marking every epoch with sleep labels
    """

    df_id = pd.read_csv("./data/aggregate/aggregate_{}.csv".format(studyId))
    # sleep_labels = df_id['sleepLabel'].values
    # df_id = df_id.drop(["sleepLabel"], axis = 1)
    df_id.dropna(thresh=df_id.shape[1], inplace=True)
    df_id = df_id.reset_index(drop=True)
    colnames = list(df_id.columns)
    df_id['HRmean'] = df_id[colnames[0:60]].mean(axis=1)
    df_id['HRstd'] = df_id[colnames[0:60]].std(axis=1)
    return df_id

def sleep_labelling(df_id, sleep_labels):
    """label sleep/wake on the aggregated dataset
    """
    sleep_labels_old = copy.deepcopy(sleep_labels)
    end_time_stamps = df_id['endTimeStamp']
    awake_range = 4*3600  # 4 hours in seconds
    awake_periods = int(awake_range/900)
    # 0 means not measured
    # -1 means not sleeping
    for i in range(1, len(sleep_labels)-1):
        # sleep starts
        if sleep_labels[i-1] == 0 and sleep_labels[i] == 0 and sleep_labels[i+1] == 1:
            current_end_stamp = end_time_stamps[i]
            for j in range(awake_periods):
                if i-j < 0:
                    break
                elif end_time_stamps[i-j] > current_end_stamp - awake_range:
                    sleep_labels[i-j] = -1
        # sleep ends
        elif sleep_labels[i-1] == 1 and sleep_labels[i] == 0 and sleep_labels[i+1] == 0:
            current_end_stamp = end_time_stamps[i]
            for j in range(awake_periods):
                if i+j >= len(sleep_labels):
                    break
                elif end_time_stamps[i+j] < current_end_stamp + awake_range:
                    sleep_labels[i+j] = -1
    sleepLabels = np.array(sleep_labels, dtype='int8')
    df_id["sleepLabel"] = sleepLabels
    df_id_labelled = df_id[df_id["sleepLabel"] != 0]
    return df_id_labelled

def add_features(hr_df, X):
    """add features to the original heart rate dataframe 
    """
    list_HRdiff_mean = []
    list_HRdiff_var = []

    for i in range(X.shape[0]):
        hr_array = X.iloc[i,:].values.flatten()
        hr_array_shifted = np.pad(hr_array[1:], (0, 1), 'constant')
        hr_diff = hr_array_shifted - hr_array
        list_HRdiff_mean.append(np.mean(hr_diff))
        list_HRdiff_var.append(np.var(hr_diff))

    X["HRdiff_mean"] = list_HRdiff_mean
    X["HRdiff_var"] = list_HRdiff_var
    return X

def extract_Xy(sid):
    """Extract features and labels from garmin data from one subject
    """
    df_id = read_aggregate(sid)
    n = df_id.shape[0]
    cols = df_id.columns.tolist()
    all_hr_id = df_id[cols[:60]]
    mean_id = np.mean(all_hr_id.values.flatten())
    var_id = np.var(all_hr_id.values.flatten())
    df_id["HRmean_deviation"] = df_id['HRmean']-np.repeat(mean_id, n)
    df_id["HR_min"] = all_hr_id.min(axis=1)
    df_id["HR_max"] = all_hr_id.max(axis=1)
    cols = df_id.columns.tolist()
    cols = cols[:-6]+cols[-5:]+[cols[-6]]
    df_id = df_id[cols]

    sleep_labels_id = df_id['sleepLabel']
    df_id_labelled = sleep_labelling(df_id, sleep_labels_id)

    colnames = list(df_id_labelled.columns.values)
    y = df_id_labelled["sleepLabel"].copy()
    X = df_id_labelled.drop(colnames[0:61]+['sleepLabel', 
                                    'steps', 
                                    'distanceInMeters', 
                                    'studyId', 
                                    'startDateTime',
                                    'startTimeStamp',
                                    'endTimeStamp'], axis = 1).copy()                                

    X['HRstd'] = X['HRstd']**2
    X = X.loc[:, ['maxMotionIntensity', 
                'meanMotionIntensity', 
                'HRstd', 
                'HR_min',
                'HR_max', 
                'HRmean', 
                'HRmean_deviation']]
    # X = add_features(all_hr_id, X)
    # X = sklearn.preprocessing.normalize(X, axis=0)

    y[y<0]=0
    y = y.to_numpy()
    y=y.astype('int')

    return X, y


def aggregate_feature_transform(aggre_df):
    """transform aggregate features dataframe
    """
    n = aggre_df.shape[0]
    cols = aggre_df.columns.tolist()
    all_hr_id = aggre_df[cols[:60]]
    mean_id = np.mean(all_hr_id.values.flatten())
    var_id = np.var(all_hr_id.values.flatten())
    aggre_df["HRmean_deviation"] = aggre_df['HRmean']-np.repeat(mean_id, n)
    aggre_df["HR_min"] = all_hr_id.min(axis=1)
    aggre_df["HR_max"] = all_hr_id.max(axis=1)

    cols = aggre_df.columns.tolist()
    cols = cols[:-6]+cols[-5:]+[cols[-6]]
    aggre_df = aggre_df[cols]
    colnames = list(aggre_df.columns.values)
    X = aggre_df.copy().drop(colnames[0:61]+['sleepLabel', 
                                            'steps', 
                                            'distanceInMeters', 
                                            'studyId', 
                                            'startDateTime',
                                            'startTimeStamp',
                                            'endTimeStamp'],
                            axis = 1)
    X['HRstd'] = X['HRstd']**2
    X = X.loc[:, ['maxMotionIntensity', 
                    'meanMotionIntensity', 
                    'HRstd', 
                    'HR_min',
                    'HR_max', 
                    'HRmean', 
                    'HRmean_deviation']]
    # X = add_features(all_hr_id, X)
    # X = sklearn.preprocessing.normalize(X, axis=0)
    return X

def preprocess_CRP(CRP, cutoff_day=100, original=False):
    """preprocess for each subject,
        CRP is the Garmin dataframe to be preprocessed
    """
    if original == False:
        small_CRP = CRP.loc[:, ["startDateTime",
                                "steps",
                                "maxMotionIntensity",
                                "meanMotionIntensity",
                                "HRstd",
                                "HR_min",
                                "HR_max",
                                "HRmean",
                                "HRmean_deviation",
                                "sleepLabel",
                                "predictedSleep"
                                ]
                            ]
    else:
        small_CRP = CRP.loc[:, ["startDateTime",
                                "steps",
                                "maxMotionIntensity",
                                "meanMotionIntensity",
                                "HRstd",
                                "HRmean",
                                "sleepLabel"
                                ]
                            ]

    small_CRP["start_date_time"] = pd.to_datetime(small_CRP["startDateTime"])
    small_CRP["Date"] = small_CRP["start_date_time"].dt.date

    small_CRP["date_string"] = small_CRP["start_date_time"].dt.strftime(
        "%Y-%m-%d")
    small_CRP["Hour"] = small_CRP["start_date_time"].dt.hour
    small_CRP["Minute"] = small_CRP["start_date_time"].dt.minute

    date_list = small_CRP["date_string"].unique()
    date_dict = {date_list[i]: i for i in range(len(date_list))}
    small_CRP["day_number"] = small_CRP["date_string"].map(date_dict)

    # small_CRP001["r"] = [0.5 if x < datetime.date(2017,3,7) else 1 for x in small_CRP001["Date"]]
    small_CRP["Time_In_Day"] = small_CRP["Hour"] + small_CRP["Minute"]/60

    if cutoff_day <= small_CRP["day_number"].max():
        small_CRP = small_CRP.loc[small_CRP["day_number"] <= cutoff_day, :]
        return small_CRP
    else:
        return small_CRP

    return small_CRP
