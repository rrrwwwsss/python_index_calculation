import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import pandas as pd
pd.set_option('display.max_columns', 1000)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 1000)
# # 设置随机种子，保证结果可重复
# np.random.seed(42)
#
# # 特征列 (100 行，4 列)
# n_samples = 100
# Feature1 = np.random.rand(n_samples) * 10  # 范围 0-10
# Feature2 = np.random.rand(n_samples) * 20  # 范围 0-20
# Feature3 = np.random.rand(n_samples) * 50  # 范围 0-50
# Feature4 = np.random.rand(n_samples) * 100 # 范围 0-100
#
# # 回归目标 (带噪声的线性组合)  回归目标是一个连续值，模型的任务是预测目标变量的具体数值。
# RegressionTarget = 3 * Feature1 - 2 * Feature2 + 0.5 * Feature3 + 10 + np.random.normal(0, 5, n_samples)
#
# # 分类目标 (根据特征简单分类)  分类目标是一个离散的类别，模型的任务是将输入样本分配到某个类别中。
# ClassificationLabel = (Feature1 + Feature2 > 15).astype(int)  # 大于 15 的分为 1 类，其余为 0 类
#
# # 创建 DataFrame
# data = pd.DataFrame({
#     "Feature1": Feature1,
#     "Feature2": Feature2,
#     "Feature3": Feature3,
#     "Feature4": Feature4,
#     "RegressionTarget": RegressionTarget,
#     "ClassificationLabel": ClassificationLabel
# })
input_file_path = r"F:/xiangmu/交通指标计算/程师兄数据处理/指标总结.csv"
# 读取地铁标识数据
data = pd.read_csv(input_file_path).head(5)
# 打印数据集
print("数据集示例：")
print(data.head())
# 特征和目标
# X = data[["抵达城市中心的直线距离", "站点间平均距离", "点度中心性/度中心性", "中介/介数中心性", "接近/临近中心性", "特征向量中心性", "PageRank", "轨道交通换乘线路","公交站台数量",'公交线路数量','路网密度','停车场密度']]
X = data[["轨道交通换乘线路"]]
y_regression = data["工作日区域内平均小时地铁出站量5.5-5.10"]
# 数据集划分
X_train, X_test, y_train, y_test = train_test_split(X, y_regression, test_size=0.2, random_state=42)

# 创建 DMatrix 对象
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

# 设置参数
params = {
    'objective': 'reg:squarederror',  # 回归任务
    'max_depth': 3, # 定义最大树深度
    'learning_rate': 0.1, #控制更新期间的步长收缩
    'n_estimators': 100 #指定树的数量
}

# 训练模型
model = xgb.train(params, dtrain, num_boost_round=100)

# 预测
y_pred = model.predict(dtest)

# 计算回归任务的性能评估指标
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"均方误差 (MSE): {mse:.4f}")
print(f"均方根误差 (RMSE): {rmse:.4f}")
print(f"平均绝对误差 (MAE): {mae:.4f}")
print(f"R² (决定系数): {r2:.4f}")

# 越小越好：MSE、RMSE 和 MAE。
# 越大越好：R²（决定系数）。

