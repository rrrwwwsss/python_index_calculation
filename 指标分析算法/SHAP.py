import matplotlib
import shap
import xgboost as xgb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes
from sklearn.metrics import mean_squared_error
# matplotlib.use('TkAgg')
matplotlib.use('Agg')  # 使用非交互式后端，避免弹出图像窗口
import os
import chardet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
pd.set_option('display.max_columns', 1000)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 1000)
# 1️⃣ 加载数据（这里用糖尿病数据集示例）
folder_path = "../结果/站点指标"  # 替换为实际路径
csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
def detect_encoding(file_path):
    with open(file_path, 'rb') as f:
        result = chardet.detect(f.read(10000))  # 读取部分内容检测编码
    return result['encoding']

df_list = [pd.read_csv(os.path.join(folder_path, file), encoding=detect_encoding(os.path.join(folder_path, file))) for file in csv_files]

# 逐个合并
merged_df = df_list[0]
for df1 in df_list[1:]:
    merged_df = pd.merge(merged_df, df1, on="name", how="outer",suffixes=('', '_drop'))
    # 删除合并后带后缀的 NAME_1 列
    merged_df.drop(columns=[col for col in merged_df.columns if col.endswith('_drop')], inplace=True)
try:
    merged_df['Parking lot density'] = pd.to_numeric(merged_df['Parking lot density'], errors='coerce')
except Exception:
    # 错误发生时什么都不做，跳过
    pass
qvyv = "D1"
shijian ="Labour Day"
hang = 3
test_size = 0.8
random_state =16
n_estimators = 100
learning_rate = 0.15
max_depth = 10
df = merged_df.loc[merged_df["NAME_1"] == qvyv].drop(columns=["NAME_1", "name"])
df = df.fillna(-1)
# df.to_csv('../指标总结/D1Test.csv', index=False, encoding='utf-8')
# 提取第一列作为因变量（注意：pandas 中的列索引从 0 开始）
# y = df.iloc[:, 1]
X = df.drop(columns=["Labour Day","Have to work","workday","weekend"])
y = df[shijian]
# 对部分列归一化处理
# X['地铁全天出站量'] = X['地铁全天出站量'].apply(lambda x: (x - X['地铁全天出站量'].min()) / (X['地铁全天出站量'].max() - X['地铁全天出站量'].min()))
# X['地铁全天进站量'] = X['地铁全天进站量'].apply(lambda x: (x - X['地铁全天进站量'].min()) / (X['地铁全天进站量'].max() - X['地铁全天进站量'].min()))
# X['公交出站客流量'] = X['公交出站客流量'].apply(lambda x: (x - X['公交出站客流量'].min()) / (X['公交出站客流量'].max() - X['公交出站客流量'].min()))
# X['公交进站客流量'] = X['公交进站客流量'].apply(lambda x: (x - X['公交进站客流量'].min()) / (X['公交进站客流量'].max() - X['公交进站客流量'].min()))

# 查看提取结果
print("因变量 y 的前 5 行：")
print(y.head())
print("\n自变量 X 的形状：", X.shape)
print(X.head())

# 将数据分为 80% 训练集和 20% 测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size,
                                                    random_state = random_state
                                                    )
# 训练 XGBoost 模型
model = xgb.XGBRegressor(objective='reg:squarederror',
n_estimators = n_estimators,  # 例如设置为 100 棵树
learning_rate = learning_rate,  # 学习率设置为 0.1
max_depth = max_depth,  # 每棵树的最大深度为 3
# n_estimators=100,  # 例如设置为 100 棵树
# learning_rate=0.15,  # 学习率设置为 0.1
# max_depth=10,  # 每棵树的最大深度为 3
subsample = 0.8,  # 每棵树随机采样 80% 的特征
colsample_bytree = 0.8,  # 每棵树随机采样 80% 的特征
min_child_weight = 0.5,
# random_state = 45
)
model.fit(X_train, y_train)
#  预测并计算误差
y_pred = model.predict(X_test)
# model = xgb.XGBRegressor(
#     n_estimators=100,
#     max_depth=5,
#     learning_rate=0.1,
#     random_state=42
# )
# model.fit(X_train, y_train)

print(f"均方误差 (MSE): {mean_squared_error(y_test, y_pred)}")
def compute_metrics(y_true, y_pred):
    """
    计算并返回回归模型的评估指标：平均绝对误差（MAE）、均方根误差（RMSE）和R²分数。

    参数:
    y_true (array-like): 实际的目标值。
    y_pred (array-like): 预测的目标值。

    返回:
    tuple: 包含三个评估指标的元组。
    """
    # 计算平均绝对误差
    mae = mean_absolute_error(y_true, y_pred)
    # 计算均方根误差
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    # 计算R²分数
    r2 = r2_score(y_true, y_pred)
    return mae, rmse, r2
mae_xgb, rmse_xgb, r2_xgb = compute_metrics(y_test, y_pred)
print("XGBoost:    MAE = {:.4f}, RMSE = {:.4f}, R2 = {:.4f}".format(mae_xgb, rmse_xgb, r2_xgb))
# 5️⃣ 使用 SHAP 解释模型
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体（SimHei）
plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示问题
explainer = shap.Explainer(model, X_train)
shap_values = explainer(X_test)
# shap_values = shap_values.abs()
print("SHAP模型解释", shap_values.values)
# 打印每个特征的 SHAP 值
shap_data = {}
for feature, shap_value in zip(X_test, np.abs(shap_values.values).mean(axis=0)):
    shap_data[feature] = shap_value
    print(f"特征: {feature}, SHAP 值: {shap_value}")
# import sys
# sys.path.append('../public')
# from 写入表2 import seve_file
# seve_file(shap_data, hang)
print("已写入")
# 6️⃣ 绘制 SHAP 分析图
# 文件夹路径
folder_path = "../结果/SHAP指标分析/"+qvyv+"_"+shijian

# 检查是否存在并创建
if not os.path.exists(folder_path):
    os.makedirs(folder_path)
    print(f"文件夹已创建：{folder_path}")
else:
    print(f"文件夹已存在：{folder_path}")
plt.figure(figsize=(1600, 1600))

# --- 全局特征重要性图 ---
plt.subplot(1, 2, 1)
shap.summary_plot(shap_values, X_test,plot_type="bar", max_display=30)
# shap.plots.bar(shap_values, max_display=30)
plt.title("特征重要性SHAP值")

# --- 局部样本解释图 (第一个样本) ---
plt.subplot(1, 2, 2)
shap.plots.beeswarm(shap_values, max_display=30)
# shap.plots.waterfall(shap_values[0])

plt.title("特征密度散点图")
# 调整两个子图之间的水平间距，wspace 的值可以根据需要调大
plt.subplots_adjust(left=0.05, right=0.98, top=0.95, bottom=0.05,wspace=1)
# plt.tight_layout()
# plt.show()
plt.savefig(folder_path+"/"+"SHAP重要性.png", dpi=300, bbox_inches="tight")
plt.close()  # 关闭图像，防止内存占用
# 批量生成并保存多张图
X_name = X_test.columns.tolist()
# # 将数据分为 80% 训练集和 20% 测试集
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1,
#                                                     random_state = random_state
#                                                     )
for i in X_name:
    shap.dependence_plot(i, shap_values.values, X_test)
    plt.savefig(folder_path+"/"+i+'.png', dpi=300, bbox_inches='tight')
    plt.close()  # 关闭图像，防止内存占用


