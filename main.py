import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from datetime import datetime, timedelta

# 构造训练数据并保存为training_data.csv
start_date = datetime(2022, 1, 1)
end_date = datetime(2023, 1, 1)
dates = pd.date_range(start=start_date, end=end_date, freq='D')

# 阈值
amplitude_threshold = 0.03
current_threshold = 2.8
speed_threshold = 1050

data = {
    '日期': [],
    '振幅': [],
    '电流': [],
    '转速': [],
    '机器告警': []
}

for date in dates:
    data['日期'].append(date)
    data['振幅'].append(amplitude_threshold)
    data['电流'].append(current_threshold)
    data['转速'].append(speed_threshold)

    if date > datetime(2023, 1, 1):
        data['机器告警'].append(1)
    else:
        data['机器告警'].append(0)

df = pd.DataFrame(data)
df.to_csv('training_data.csv', index=False)

# 读取训练数据
data = pd.read_csv('training_data.csv')
X = data[['振幅', '电流', '转速']]
y = data['机器告警']

# 构建混合模型（随机森林和支持向量机）
rf_model = RandomForestClassifier()
svm_model = SVC()

# 训练模型
rf_model.fit(X, y)
svm_model.fit(X, y)

# 预测2023-01-01之后可能发生机器告警的日期
future_dates = pd.date_range(start=datetime(2023, 1, 2), end=datetime(2024, 12, 31), freq='D')
future_data = {
    '日期': future_dates,
    '振幅': [amplitude_threshold] * len(future_dates),
    '电流': [current_threshold] * len(future_dates),
    '转速': [speed_threshold] * len(future_dates)
}

future_df = pd.DataFrame(future_data)
future_X = future_df[['振幅', '电流', '转速']]

# 使用混合模型进行预测
rf_pred = rf_model.predict(future_X)
svm_pred = svm_model.predict(future_X)

# 输出预测机器告警的日期
predicted_dates = future_dates[rf_pred == 1]  # 选择随机森林模型预测结果为1的日期

print(predicted_dates)
