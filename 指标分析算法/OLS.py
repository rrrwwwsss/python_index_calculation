import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 读取数据
df_all = pd.read_excel(r'../指标总结/CPNT.xls')
df = df_all.iloc[0:3]
# 定义自变量（X）和因变量（y）
X = df.iloc[:, 5:27]  # 所有自变量
X = sm.add_constant(X)  # 添加截距项
y = df.iloc[:, 1]
print(X)
print(y)
# 拟合OLS模型
ols_model = sm.OLS(y, X).fit()

# 获取预测值
y_pred = ols_model.predict(X)
print(y_pred)
# 计算MAE、RMSE和R²
mae = mean_absolute_error(y, y_pred)
rmse = np.sqrt(mean_squared_error(y, y_pred))
r2 = ols_model.rsquared

# 输出结果
print(f"MAE: {mae}")
print(f"RMSE: {rmse}")
print(f"R²: {r2}")