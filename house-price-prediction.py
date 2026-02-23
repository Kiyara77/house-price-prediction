#导入需要用到的库
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from xgboost import XGBRegressor
train = pd.read_csv('D:\\1\\train.csv') #读取train文件
test = pd.read_csv('D:\\1\\test.csv') #读取test文件
test_ids = test['Id'] #结果中需包含Id列
all_data = pd.concat([train.drop('SalePrice', axis=1), test], ignore_index=True) #去除SalePrice列后合并，便于进行数据清洗
numeric_features = all_data.select_dtypes(include=[np.number]).columns #提取数值型数据
all_data[numeric_features] = all_data[numeric_features].fillna(all_data[numeric_features].median()) #用中位数填充缺失值
categorical_features = all_data.select_dtypes(exclude=[np.number]).columns #提取分类变量
all_data[categorical_features] = all_data[categorical_features].fillna(all_data[categorical_features].mode().iloc[0]) #用众数填充缺失值
for col in categorical_features:
    le = LabelEncoder()  #对分类变量进行标签编码
    all_data[col] = le.fit_transform(all_data[col].astype(str))
X_train = all_data.iloc[:len(train)] #拆分训练集和测试集
X_test = all_data.iloc[len(train):]
y_train = train['SalePrice']
model = XGBRegressor(  #定义并训练XGBoost模型
    n_estimators=1000, #树的数量
    max_depth=6,  #每棵树的最大深度
    learning_rate=0.05, #学习率
    random_state=42 #随机种子
)
model.fit(X_train, y_train)
predictions = model.predict(X_test)  #对测试集进行预测
submission = pd.DataFrame({   #生成结果
    'Id': test_ids,
    'SalePrice': predictions
})
submission.to_csv('D:\\1\\submission.csv', index=False)
print(submission.head())
y_train_log = np.log(y_train)
model = XGBRegressor( #定义XGBoost模型
    n_estimators=1000,
    max_depth=6,
    learning_rate=0.05,
    random_state=42
)
#计算负均方误差
cv_scores = -cross_val_score(model, X_train, y_train_log, 
                             cv=5, scoring='neg_mean_squared_error')
#转换为 RMSE
rmse_scores = np.sqrt(cv_scores)
print(f"5-Fold CV RMSE: {rmse_scores.mean():.5f} ± {rmse_scores.std():.5f}")
