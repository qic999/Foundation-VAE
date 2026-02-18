import random
import os

# 设定文件路径
train_file = 'train.txt'  # 训练集输出文件
eval_file = 'eval.txt'    # 验证集输出文件
test_file = 'test.txt'    # 测试集输出文件

# 读取数据
data = sorted(os.listdir('/storage/chenqi/data/BraTS_2019_Data_Training/All'))

# 随机打乱数据
random.shuffle(data)

# 计算各个数据集的大小
train_size = 290
eval_size = 8
test_size = 37

# 划分数据集
train_data = data[:train_size]
eval_data = data[train_size:train_size + eval_size]
test_data = data[train_size + eval_size:]

# 保存到txt文件
with open(train_file, 'w') as file:
    for i in train_data:
        file.write(i)
        file.write('\n')

with open(eval_file, 'w') as file:
    for i in eval_data:
        file.write(i)
        file.write('\n')


with open(test_file, 'w') as file:
    for i in test_data:
        file.write(i)
        file.write('\n')


print(f"数据集已划分完成，并分别保存为: {train_file}, {eval_file}, {test_file}")
