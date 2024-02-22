#Decision tree no time

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

def read_and_combine_json_in_subfolders(directory):
    combined_df = pd.DataFrame()

    # 讀指定目錄下的所有子目錄與文件
    for subdir, dirs, files in os.walk(directory):
        for file in files:
            # 檢查文件是否为 JSON 文件
            if file.endswith('.json'):
                file_path = os.path.join(subdir, file)
                # 讀取 JSON 文件
                df = pd.read_json(file_path)
                # 將讀取的數據加到到主 DataFrame
                combined_df = pd.concat([combined_df, df], ignore_index=True)
    
    return combined_df

# 指定包含 JSON 文件的文件夾
directory = './20231203/merge_iou50_mode_3_7/transformtime'  

# 讀取並合併數據
combined_data = read_and_combine_json_in_subfolders(directory)


# 提取特徵和目標變量
X = combined_data.iloc[:, [2, 4]]  # 第3列和第5列作為特徵
y = combined_data.iloc[:, 5]       # 第6列作為目標
''''''
# 計算第六列中0, 1, 2, 3出現的次數
#value_counts = combined_data.iloc[:, 5].value_counts()
#print(value_counts)

# 分割數據集為訓練集和測試集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 創建決策樹分類器實例
clf = DecisionTreeClassifier()


# 訓練決策樹
clf.fit(X_train, y_train)

# 在測試集上進行預測
y_pred = clf.predict(X_test)

# 計算準確率
accuracy = accuracy_score(y_test, y_pred)
print(f" Accuracy: {accuracy}")



from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO  
from IPython.display import Image  
import pydotplus

combined_data = StringIO()
export_graphviz(clf, out_file=combined_data ,  
                filled=True, rounded=True,
                special_characters=True, feature_names = feature_cols,class_names=['0','1','2','3'])
graph = pydotplus.graph_from_dot_data(combined_data .getvalue())  
graph.write_png('diabetes.png')
Image(graph.create_png())