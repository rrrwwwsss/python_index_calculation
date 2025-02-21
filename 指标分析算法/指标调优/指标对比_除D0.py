import numpy as np
import statsmodels.api as sm
from sklearn.linear_model import HuberRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import os
import chardet
pd.set_option('display.max_columns', 1000)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 1000)
def zhibiao(test_size,random_state,qvyv,time):
    # ------------------------------
    # 1. 数据构造与划分
    # ------------------------------
    # 读取 Excel 文件（xls 格式）
    print(test_size)
    print(random_state)
    folder_path = "../../结果/站点指标"  # 替换为实际路径
    csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
    def detect_encoding(file_path):
        with open(file_path, 'rb') as f:
            result = chardet.detect(f.read(10000))  # 读取部分内容检测编码
        return result['encoding']

    df_list = [pd.read_csv(os.path.join(folder_path, file), encoding=detect_encoding(os.path.join(folder_path, file))) for file in csv_files]
    # print(df_list)
    # 逐个合并
    merged_df = df_list[0]
    for df1 in df_list[1:]:
        merged_df = pd.merge(merged_df, df1, on="name", how="outer",suffixes=('', '_drop'))
        # 删除合并后带后缀的 NAME_1 列
        merged_df.drop(columns=[col for col in merged_df.columns if col.endswith('_drop')], inplace=True)
    try:
        merged_df['Parking Density'] = pd.to_numeric(merged_df['Parking Density'], errors='coerce')
    except Exception:
        # 错误发生时什么都不做，跳过
        pass
    # print(merged_df)
    # merged_df.to_csv('../指标总结/AllTest.csv', index=False, encoding='utf-8')
    # df = merged_df.loc[merged_df["NAME_1"] == qvyv].drop(columns=["NAME_1", "name"])
    df = merged_df[~merged_df['NAME_1'].isin([qvyv])].drop(columns=["NAME_1", "name"])
    print(df)
    df = df.fillna(-1)
    # df.to_csv('../指标总结/D1Test.csv', index=False, encoding='utf-8')
    # 提取第一列作为因变量（注意：pandas 中的列索引从 0 开始）
    # y = df.iloc[:, 1]
    X = df.drop(columns=["Labour Day","Have to work","workday","weekend"])  # 取除 count 以外的列
    # 提取第 5 到第 30 列作为自变量
    # 注意：Excel 中第 5 列对应 DataFrame 的第 4 列（索引 4），第 30 列对应索引 29
    # X = df.iloc[:,[4,5]+ list(range(6,27))]
    # X = df.iloc[:, 5:27]
    y = df[time]  # 取 count 列
    print(df)
    # 将数据分为 80% 训练集和 20% 测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size,
                                                        random_state = random_state
                                                        )

    # ------------------------------
    # 2. 模型拟合
    # ------------------------------

    # 2.1 OLS 模型（使用 statsmodels，需要手动添加截距项）
    X_train_sm = X_train
    # X_train_sm['const'] = 1
    # X_train_sm = sm.add_constant(X_train)
    print("训练集")
    # X_test_sm = sm.add_constant(X_test)
    X_test_sm = X_test
    # X_test_sm['const'] = 1
    print("测试集")
    ols_model = sm.OLS(y_train, X_train_sm).fit()
    y_pred_ols = ols_model.predict(X_test_sm)

    # 2.3 XGBoost 模型
    xgb_model = xgb.XGBRegressor(objective='reg:squarederror',
    n_estimators = 100,  # 例如设置为 100 棵树
    learning_rate = 0.15,  # 学习率设置为 0.1
    max_depth = 10,  # 每棵树的最大深度为 3
    subsample = 0.8,  # 每棵树随机采样 80% 的特征
    colsample_bytree = 0.8,  # 每棵树随机采样 80% 的特征
    min_child_weight = 0.5,
    # random_state = 45
    )
    xgb_model.fit(X_train, y_train)
    y_pred_xgb = xgb_model.predict(X_test)

    # ------------------------------
    # 3. 计算评价指标
    # ------------------------------
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


    mae_ols, rmse_ols, r2_ols = compute_metrics(y_test, y_pred_ols)
    mae_xgb, rmse_xgb, r2_xgb = compute_metrics(y_test, y_pred_xgb)

    print("OLS:        MAE = {:.4f}, RMSE = {:.4f}, R2 = {:.4f}".format(mae_ols, rmse_ols, r2_ols))
    print("XGBoost:    MAE = {:.4f}, RMSE = {:.4f}, R2 = {:.4f}".format(mae_xgb, rmse_xgb, r2_xgb))
    if r2_xgb < 0.1:
        return test_size,random_state,0,0
    # ------------------------------
    # 4. 计算改进率函数
    # ------------------------------
    def improvement_percentage(baseline, new, higher_better=False):
        """
        若 higher_better 为 False（如 MAE、RMSE），则越小越好，改进率 = ((baseline - new) / baseline) * 100；
        若 higher_better 为 True（如 R2），则越大越好，改进率 = ((new - baseline) / baseline) * 100；
        若 baseline 为 0，则返回 np.nan。
        """
        if baseline == 0:
            return np.nan
        if higher_better:
            return ((new - baseline) / abs(baseline)) * 100
        else:
            return ((baseline - new) / abs(baseline)) * 100

    # ------------------------------
    # 5. 计算 XGBoost 相对于 OLS 的改进率
    # ------------------------------
    print("\nImprovement of XGBoost over OLS:")
    print("MAE improvement:   {:.2f}%".format(improvement_percentage(mae_ols, mae_xgb, higher_better=False)))
    print("RMSE improvement:  {:.2f}%".format(improvement_percentage(rmse_ols, rmse_xgb, higher_better=False)))
    print("R2 improvement:    {:.2f}%".format(improvement_percentage(r2_ols, r2_xgb, higher_better=True)))

    all_data = [[mae_ols, rmse_ols, r2_ols],[mae_xgb, rmse_xgb, r2_xgb],[improvement_percentage(mae_ols, mae_xgb, higher_better=False),improvement_percentage(rmse_ols, rmse_xgb, higher_better=False),improvement_percentage(r2_ols, r2_xgb, higher_better=True)]]
    return test_size,random_state,r2_xgb,all_data

