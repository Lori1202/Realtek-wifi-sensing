#Decision tree time
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

def read_and_combine_json_in_subfolders(directory):
    combined_df = pd.DataFrame()

    for subdir, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.json'):
                file_path = os.path.join(subdir, file)
                df = pd.read_json(file_path)
                combined_df = pd.concat([combined_df, df], ignore_index=True)
    
    return combined_df

# 將日期時間字符串轉換為 Unix 時間戳（秒數）
def convert_to_timestamp(df, column_name):
    df[column_name] = pd.to_datetime(df[column_name])
    df[column_name] = (df[column_name] - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')
    return df

directory = './20231203/merge_iou50_mode_3_7/transformtime'  
combined_data = read_and_combine_json_in_subfolders(directory)

# 轉換時間列
combined_data = convert_to_timestamp(combined_data, combined_data.columns[0])

# 提取特徵和目標變量
X = combined_data.iloc[:, [0, 2, 4]]  # 第1列、第3列和第5列作為特徵
y = combined_data.iloc[:, 5]          # 第6列作為目標

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=50)
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")
