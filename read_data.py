import pandas as pd
import numpy as np
import random
from pathlib import Path

MIN_NOISE_FROM_PREV = 0.1  # In paper: 0.4
MAX_MOISE_FROM_PREV = 0.2  # In paper: 0.8
MIN_NOISE_INTERVAL = 3  # In paper: 5f
MAX_NOISE_INTERVAL = 7  # In paper: 12


def read_data(path):
    # Read data and set flag 0 to all correct rows
    col_list = ['angle', 'torque', 'speed']
    csv_data = pd.read_csv(path, usecols=col_list)

    csv_data = csv_data.drop_duplicates(subset=['angle'])
    csv_data = csv_data.drop_duplicates(subset=['torque'])
    csv_data = csv_data.drop_duplicates(subset=['speed'])

    csv_data['flag'] = 0
    return csv_data


def add_noise(csv_data):
    random_interval = random.randint(MIN_NOISE_INTERVAL, MAX_NOISE_INTERVAL)
    i = 0
    index_to_add = random_interval

    while i <= csv_data.shape[0]:
        if i == index_to_add:
            prev_angle = csv_data.iloc[index_to_add - 1]['angle']
            prev_torque = csv_data.iloc[index_to_add - 1]['torque']
            prev_speed = csv_data.iloc[index_to_add - 1]['speed']

            new_angle = random.choice([np.random.uniform(prev_angle - MAX_MOISE_FROM_PREV, prev_angle - MIN_NOISE_FROM_PREV),
                                       np.random.uniform(prev_angle + MIN_NOISE_FROM_PREV, prev_angle + MAX_MOISE_FROM_PREV)])
            new_torque = random.choice([np.random.uniform(prev_torque - MAX_MOISE_FROM_PREV, prev_torque - MIN_NOISE_FROM_PREV),
                                        np.random.uniform(prev_torque + MIN_NOISE_FROM_PREV, prev_torque + MAX_MOISE_FROM_PREV)])
            new_speed = random.choice([np.random.uniform(prev_speed - MAX_MOISE_FROM_PREV, prev_speed - MIN_NOISE_FROM_PREV),
                                       np.random.uniform(prev_speed + MIN_NOISE_FROM_PREV, prev_speed + MAX_MOISE_FROM_PREV)])
            new_line = pd.DataFrame([{'angle': new_angle, 'torque': new_torque, 'speed': new_speed, 'flag': 1}],
                                    index=[index_to_add])
            csv_data = pd.concat([csv_data.iloc[:index_to_add], new_line, csv_data.iloc[index_to_add:]]).reset_index(
                drop=True)
            random_interval = random.randint(MIN_NOISE_INTERVAL, MAX_NOISE_INTERVAL)
            index_to_add = random_interval + i
        i = i + 1
    return csv_data


def save_data(csv_data, new_path):
    filepath = Path(new_path)
    csv_data = csv_data.reset_index(drop=True)
    csv_data.to_csv(filepath)


data1_from = './output/dataset/interpolated.csv'
data1_to = './data-noise-added/data1_with_noise.csv'
data2_from = './output/dataset-2-2/interpolated.csv'
data2_to = './data-noise-added/data2_with_noise.csv'


data1 = read_data(data1_from)
#data1 = add_noise(data1)
save_data(data1, data1_to)

data2 = read_data(data2_from)
#data2 = add_noise(data2)
save_data(data2, data2_to)
