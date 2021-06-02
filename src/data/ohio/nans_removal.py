def remove_nans(data):
    """
    Remove samples that still have NaNs (in y column mostly)
    :param data: dataframe of samples
    :return: no-NaN dataframe
    """
    new_data = []
    for df in data:
        new_data.append(df.dropna())
    return new_data