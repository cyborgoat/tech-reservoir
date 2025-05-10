import pandas as pd
import random
from datetime import datetime, timedelta

# Define the schema
schema = ['numeric_1', 'numeric_2', 'numeric_3', '客户类型', '国际', '是否私行客户', 'date']

# Define the countries
countries = ['中国', '美国', '日本', '印度', '巴西', '俄罗斯', '德国', '法国', '加拿大', '澳大利亚']

# Define the customer types
customer_types = ['私人客户', '机构客户']

# Define a function to generate a random date within a given range
def random_date(start_date, end_date):
    return start_date + timedelta(
        seconds=random.randint(0, int((end_date - start_date).total_seconds())))

# Define the start and end dates for the random date generator
start_date = datetime.strptime('2020/01/01', '%Y/%m/%d')
end_date = datetime.strptime('2023/12/31', '%Y/%m/%d')

# Define a function to generate outlier or normal numeric values
def generate_value(outlier_prob=0.05, outlier_scale=10):
    if random.random() < outlier_prob:
        return random.randint(100, 1000) * outlier_scale  # Generate outlier
    else:
        return random.randint(100, 1000)  # Generate normal value

# Generate the data
data = []
for _ in range(1000):
    row = [
        generate_value() if random.random() > 0.05 else None,  # numeric_1
        generate_value() if random.random() > 0.05 else None,  # numeric_2
        generate_value() if random.random() > 0.05 else None,  # numeric_3
        random.choice(customer_types) if random.random() > 0.05 else None,  # 客户类型
        random.choice(countries) if random.random() > 0.05 else None,  # 国际
        random.randint(0, 1) if random.random() > 0.05 else None,  # 是否私行客户
        random_date(start_date, end_date) if random.random() > 0.05 else None  # date
    ]
    data.append(row)

# Create a dataframe
df = pd.DataFrame(data, columns=schema)

# Replace None with NaN
df.fillna(value=pd.np.nan, inplace=True)

# Write the dataframe to a .csv file
df.to_csv('data.csv', index=False)
