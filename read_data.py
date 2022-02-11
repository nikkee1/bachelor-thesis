import pandas as pd
import numpy as np
import random
from pathlib import Path


def read_data(path):
    # Read data and set flag 0 to all correct rows
    col_list = ['angle', 'torque', 'speed']
    csv_data = pd.read_csv(path, usecols=col_list)
    csv_data['flag'] = 0
    return csv_data


def add_noise(csv_data):
    random_interval = random.randint(5, 12)
    i = 0
    index_to_add = random_interval

    while i <= csv_data.shape[0]:
        if i == index_to_add:
            prev_angle = csv_data.iloc[index_to_add - 1]['angle']
            prev_torque = csv_data.iloc[index_to_add - 1]['torque']
            prev_speed = csv_data.iloc[index_to_add - 1]['speed']

            new_angle = random.choice([np.random.uniform(prev_angle - 0.8, prev_angle - 0.4),
                                       np.random.uniform(prev_angle + 0.4, prev_angle + 0.8)])
            new_torque = random.choice([np.random.uniform(prev_torque - 0.8, prev_torque - 0.4),
                                        np.random.uniform(prev_torque + 0.4, prev_torque + 0.8)])
            new_speed = random.choice([np.random.uniform(prev_speed - 0.8, prev_speed - 0.4),
                                       np.random.uniform(prev_speed + 0.4, prev_speed + 0.8)])
            new_line = pd.DataFrame([{'angle': new_angle, 'torque': new_torque, 'speed': new_speed, 'flag': 1}],
                                    index=[index_to_add])
            csv_data = pd.concat([csv_data.iloc[:index_to_add], new_line, csv_data.iloc[index_to_add:]]).reset_index(
                drop=True)
            random_interval = random.randint(5, 12)
            index_to_add = random_interval + i
        i = i + 1
    return csv_data


def save_data(csv_data, new_path):
    filepath = Path(new_path)
    csv_data.to_csv(filepath)


data1_from = './output/dataset/interpolated.csv'
data1_to = './data-noise-added/data1_with_noise.csv'
data2_from = './output/dataset-2-2/interpolated.csv'
data2_to = './data-noise-added/data2_with_noise.csv'


data1 = read_data(data1_from)
data1 = add_noise(data1)
save_data(data1, data1_to)

data2 = read_data(data2_from)
data2 = add_noise(data2)
save_data(data2, data2_to)
