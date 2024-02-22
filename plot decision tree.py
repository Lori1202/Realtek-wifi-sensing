import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

def read_and_combine_json_in_subfolders(directory):
    combined_df = pd.DataFrame()

    for subdir, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.json'):
                file_path = os.path.join(subdir, file)
                df = pd.read_json(file_path)
                combined_df = pd.concat([combined_df, df], ignore_index=True)
    
    return combined_df

# 指定包含 JSON 文件的文件夾
directory = './20231203/merge_iou50_mode_3_7/transformtime'  

# 讀取並合併數據
combined_data = read_and_combine_json_in_subfolders(directory)

# 提取特徵和目標變量
X = combined_data.iloc[:, [2, 4]]
y = combined_data.iloc[:, 5]

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
print(f"Accuracy: {accuracy}")

# 绘制决策树
plt.figure(figsize=(10,5))
plot_tree(clf, filled=True, feature_names=['Feature 1', 'Feature 2'], class_names=['Class 0', 'Class 1', 'Class 2', 'Class 3'])
# 保存圖像到文件
plt.savefig('decision_tree.png', format='png', dpi=300, bbox_inches='tight')
plt.show()
