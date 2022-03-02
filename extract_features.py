import pandas as pd
from pathlib import Path
import random
import numpy as np

MIN_NOISE_FROM_PREV = 0.008  # In paper: 0.4
MAX_MOISE_FROM_PREV = 0.04  # In paper: 0.8
MIN_NOISE_INTERVAL = 2  # In paper: 5
MAX_NOISE_INTERVAL = 3  # In paper: 12


def read_data(path, feature):
    col_list = [feature, 'flag']
    csv_data = pd.read_csv(path, usecols=col_list)
    return csv_data


def make_delayed_data(data, feature):
    result_data = pd.DataFrame()
    for i in range(3, data.shape[0]):
        ft = data.iloc[i][feature]
        ft_1 = data.iloc[i - 1][feature]
        ft_2 = data.iloc[i - 2][feature]
        ft_3 = data.iloc[i - 3][feature]
        flag = data.iloc[i]['flag']

        '''
        if ft in result_data.values:
            continue
        '''
        new_line = pd.DataFrame([{'f(t)': ft, 'f(t-1)': ft_1, 'f(t-2)': ft_2, 'f(t-3)': ft_3, 'f(t)-f(t-1)': ft - ft_1,
                                  'f(t)-f(t-2)': ft - ft_2, 'f(t)-f(t-3)': ft - ft_3, 'flag': flag}])
        result_data = result_data.append(new_line)
    return result_data


def save_data(csv_data, dir_name, feature):
    filename = feature
    suffix = '.csv'
    filepath = Path(dir_name, filename).with_suffix(suffix)
    csv_data.to_csv(filepath, index=False)


def add_noise(csv_data, feature):
    random_interval = random.randint(MIN_NOISE_INTERVAL, MAX_NOISE_INTERVAL)
    i = 0
    index_to_add = random_interval

    while i <= csv_data.shape[0]:
        if i == index_to_add:
            previous = csv_data.iloc[index_to_add - 1][feature]

            new = random.choice([np.random.uniform(previous - MAX_MOISE_FROM_PREV, previous - MIN_NOISE_FROM_PREV),
                                       np.random.uniform(previous + MIN_NOISE_FROM_PREV, previous + MAX_MOISE_FROM_PREV)])
            new_line = pd.DataFrame([{feature: new,  'flag': 1}], index=[index_to_add])
            csv_data = pd.concat([csv_data.iloc[:index_to_add], new_line, csv_data.iloc[index_to_add:]]).reset_index(
                drop=True)
            random_interval = random.randint(MIN_NOISE_INTERVAL, MAX_NOISE_INTERVAL)
            index_to_add = random_interval + i
        i = i + 1
    return csv_data


path_to = './data-features-extracted'

data1_from = './data-noise-added/data1_with_noise.csv'
data2_from = './data-noise-added/data2_with_noise.csv'

data1_angle = read_data(data1_from, 'angle')
data2_angle = read_data(data2_from, 'angle')
tot_angle = pd.concat([data1_angle, data2_angle], ignore_index=True)
tot_angle = add_noise(tot_angle, 'angle')

data1_torque = read_data(data1_from, 'torque')
data2_torque = read_data(data2_from, 'torque')
tot_torque = pd.concat([data1_torque, data2_torque], ignore_index=True)
tot_torque = add_noise(tot_torque, 'torque')

data1_speed = read_data(data1_from, 'speed')
data2_speed = read_data(data2_from, 'speed')
tot_speed = pd.concat([data1_speed, data2_speed], ignore_index=True)
tot_speed = add_noise(tot_speed, 'speed')

angle_extracted = make_delayed_data(tot_angle, 'angle')
torque_extracted = make_delayed_data(tot_torque, 'torque')
speed_extracted = make_delayed_data(tot_speed, 'speed')

save_data(angle_extracted, path_to, 'angle')
save_data(torque_extracted, path_to, 'torque')
save_data(speed_extracted, path_to, 'speed')
