import os
import xml.etree.ElementTree as ET
from numpy.lib.twodim_base import mask_indices
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler

def load_ohio(subject, DATA_DIR='data/raw/ohio-t1dm/2018/'):
    """
    Load OhioT1DM training_old and testing files into a dataframe
    :param subject: name of subject
    :return: dataframe
    """
    train_path, test_path = _compute_file_names(subject, DATA_DIR)

    [train_xml, test_xml] = [ET.parse(set).getroot() for set in [train_path, test_path]]

    [train, test] = [_extract_data_from_xml(xml) for xml in [train_xml, test_xml]]

    data = pd.concat([train, test], ignore_index=True)

    return data


def _compute_file_names(subject, DATA_DIR):
    """
    Compute the name of the files, given dataset and subject
    :param subject: name of subject
    :return: path to training file, path to testing file
    """
    train_dir = os.path.join(DATA_DIR, "train")
    train_file = subject + "-ws-training.xml"
    train_path = os.path.join(train_dir, train_file)

    test_dir = os.path.join(DATA_DIR, "test")
    test_file = subject + "-ws-testing.xml"
    test_path = os.path.join(test_dir, test_file)

    return train_path, test_path


def _extract_data_from_xml(xml):
    """
    extract glucose, CHO, and insulin from xml and merge the data
    :param xml:
    :return: dataframe
    """
    glucose_df = _get_glucose_from_xml(xml)
    CHO_df = _get_CHO_from_xml(xml)
    insulin_df = _get_insulin_from_xml(xml)
    meal_df = _get_meal_from_xml(xml)

    df = pd.merge(glucose_df, CHO_df, how="outer", on="datetime")
    df = pd.merge(df, insulin_df, how="outer", on="datetime")
    df = pd.merge(df, meal_df, how="outer", on="datetime")
    df = df.sort_values("datetime")

    return df


def _get_field_labels(etree, field_index):
    """
    extract labels from xml tree
    :param etree: etree
    :param field_index:  position of field
    :return:
    """
    return list(etree[field_index][0].attrib.keys())


def _iter_fields(etree, field_index):
    """
    extract columns inside xml tree
    :param etree: tree
    :param field_index: position of columns
    :return:
    """
    for event in etree[field_index].iter("event"):
        yield list(event.attrib.values())


def _get_CHO_from_xml(xml):
    """
    Extract CHO values from xml
    :param xml:
    :return: CHO dataframe
    """
    labels = _get_field_labels(xml, field_index=5)
    CHO = list(_iter_fields(xml, field_index=5))
    CHO_df = pd.DataFrame(data=CHO, columns=labels)
    CHO_df.drop("type", axis=1, inplace=True)
    CHO_df["ts"] = pd.to_datetime(CHO_df["ts"], format="%d-%m-%Y %H:%M:%S")
    CHO_df["carbs"] = CHO_df["carbs"].astype("float")
    CHO_df.rename(columns={'ts': 'datetime', 'carbs': 'CHO'}, inplace=True)
    return CHO_df


def _get_insulin_from_xml(xml):
    """
    Extract insulin values from xml
    :param xml:
    :return: insulin dataframe
    """
    labels = _get_field_labels(xml, field_index=4)
    insulin = list(_iter_fields(xml, field_index=4))
    insulin_df = pd.DataFrame(data=insulin, columns=labels)
    for col in ["ts_end", "type", "bwz_carb_input"]:
        insulin_df.drop(col, axis=1, inplace=True)
    insulin_df["ts_begin"] = pd.to_datetime(insulin_df["ts_begin"], format="%d-%m-%Y %H:%M:%S")
    insulin_df["dose"] = insulin_df["dose"].astype("float")
    insulin_df.rename(columns={'ts_begin': 'datetime', 'dose': 'insulin'}, inplace=True)
    return insulin_df


def _get_glucose_from_xml(xml):
    """
    Extract glucose values from xml
    :param xml:
    :return: glucose dataframe
    """
    labels = _get_field_labels(xml, field_index=0)
    glucose = list(_iter_fields(xml, field_index=0))
    glucose_df = pd.DataFrame(data=glucose, columns=labels)
    glucose_df["ts"] = pd.to_datetime(glucose_df["ts"], format="%d-%m-%Y %H:%M:%S")
    glucose_df["value"] = glucose_df["value"].astype("float")
    glucose_df.rename(columns={'ts': 'datetime', 'value': 'glucose'}, inplace=True)
    return glucose_df

def _get_meal_from_xml(xml):
    """
    Extract meal values from xml
    :param xml:
    :return: meal dataframe
    """
    labels = _get_field_labels(xml, field_index=5)
    meal = list(_iter_fields(xml, field_index=5))
    meal_df = pd.DataFrame(data=meal, columns=labels)
    meal_df["ts"] = pd.to_datetime(meal_df["ts"], format="%d-%m-%Y %H:%M:%S")
    meal_df["type"] = meal_df["type"].astype('category')
    meal_df.rename(columns={'ts': 'datetime', 'type':'meal'}, inplace=True)
    return meal_df

def scale(train, test, hist, ph, standardize):
    """
    Normalize
    Normalize features = glucose, CHO, insulin, carbs
    Label Encoding features = Subject
    :param train: training
    :param test: testing
    :param hist: history size
    :param ph: target size
    :return: standardized training and test sets
    :return: scalers used in a dict
    """
    normalize_features = ['glucose', 'CHO', 'insulin']
    label_features = ['subject', 'meal']
    if standardize:
        scalers = {feature:StandardScaler() for feature in normalize_features}
    else:
        scalers = {feature:MinMaxScaler(feature_range=(0.1,0.9)) for feature in normalize_features}
    scalers.update({feature:LabelEncoder() for feature in label_features})

    scaled_train, scaled_test = train.copy(), test.copy()
    for feature in normalize_features+label_features:
        features = [f'{feature}_{i}' for i in range(hist)]
        scaled_train[features] = scalers[feature].fit_transform(scaled_train[features].values.reshape(-1,1))\
                                .reshape(scaled_train[features].values.shape)
        scaled_test[features] = scalers[feature].transform(scaled_test[features].values.reshape(-1,1))\
                                .reshape(scaled_test[features].values.shape)
    # Scale y with glucose scaler
    # Scale features y_1 to y_n with the glucose scaler
    for target_time in range(ph):
        target_feat = f'y_{target_time}'
        scaled_train[target_feat] = scalers['glucose'].transform(scaled_train[target_feat].values.reshape(-1,1)).flatten()
        scaled_test[target_feat] = scalers['glucose'].transform(scaled_test[target_feat].values.reshape(-1,1)).flatten()
    return scaled_train, scaled_test, scalers


if __name__ == "__main__":
    df = load_ohio("559", DATA_DIR='data/raw/ohio-t1dm/2018/')
    print(df)