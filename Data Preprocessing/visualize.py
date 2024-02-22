import os
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

def convert_unix_timestamps_in_folder(input_directory, output_directory):
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    for filename in os.listdir(input_directory):
        if filename.endswith('.csv'):
            file_path = os.path.join(input_directory, filename)
            try:
                df = pd.read_csv(file_path, header=None)
                df[0] = df[0].apply(lambda x: datetime.utcfromtimestamp(int(x)).strftime('%Y-%m-%d %H:%M:%S'))
                output_file_path = os.path.join(output_directory, filename)
                df.to_csv(output_file_path, index=False, header=False)
            except Exception as e:
                print(f"處理文件 {filename} 時發生錯誤: {e}")

def plot_and_save_csv_files(directory):
    for filename in os.listdir(directory):
        if filename.endswith('.csv'):
            file_path = os.path.join(directory, filename)
            try:
                df = pd.read_csv(file_path, header=None)
                if df.shape[1] >= 6:
                    x = df.iloc[:, 0]
                    y1 = df.iloc[:, 2]
                    y2 = df.iloc[:, 4]
                    people_changes = df.iloc[:, 5]

                    plt.figure()
                    plt.plot(x, y1, label='motion_status')
                    plt.plot(x, y2, label='micro_motion_status')

                    for i in range(1, len(people_changes)):
                        if people_changes[i] != people_changes[i-1]:
                            plt.axvline(x=x[i], color='red', linestyle='--')

                    plt.xlabel('X Axis')
                    plt.ylabel('Y Axis')
                    plt.title('Plot from ' + filename)
                    plt.xticks(rotation=90)
                    plt.ylim(-5, 110)
                    plt.legend()

                    # 建立子資料夾
                    unique_values = set(df.iloc[:, 5].unique())
                    subfolder = get_subfolder_name(unique_values)

                    subfolder_path = os.path.join(directory, subfolder)
                    if not os.path.exists(subfolder_path):
                        os.makedirs(subfolder_path)

                    plot_filename = filename.replace('.csv', '.png')
                    plt.savefig(os.path.join(subfolder_path, plot_filename))
                    plt.close()

                    new_file_path_csv = os.path.join(subfolder_path, filename)
                    new_file_path_json = new_file_path_csv.replace('.csv', '.json')
                    df.to_json(new_file_path_json, orient='records', force_ascii=False)
                    os.rename(file_path, new_file_path_csv)
                    # os.remove(new_file_path_csv) #刪除產生出的CSV
            except Exception as e:
                print(f"處理文件 {filename} 時發生錯誤 {e}")

def get_subfolder_name(unique_values):
    if unique_values == {0, 1}:
        return "01"
    elif unique_values == {1}:
        return "1"
    elif unique_values == {2}:
        return "2"
    elif unique_values == {3}:
        return "3"
    elif unique_values == {1, 2}:
        return "21"
    elif unique_values == {1, 2, 3}:
        return "321"
    elif unique_values == {1, 3}:
        return "31"
    elif unique_values == {2, 3}:
        return "32"
    else:
        return "other"

input_directory = './20231203/merge_iou50_mode_3_7/'
output_directory = './20231203/merge_iou50_mode_3_7/transformtime'               
directory = './20231203/merge_iou50_mode_3_7/transformtime'

convert_unix_timestamps_in_folder(input_directory, output_directory)
plot_and_save_csv_files(directory)
