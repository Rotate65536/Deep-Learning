from MLP import multilayerPerceptron as mP
import numpy as np
import matplotlib.pyplot as plt

inputs = np.array([[0, 0],
                   [1, 0],
                   [0, 1],
                   [1, 1]])
targets = np.array([[0],
                    [1],
                    [1],
                    [0]])

# 设置参数
epochs = 10000
learning_rate = 0.09
num_neurons = [2, 10, 1]

# 构建模型
MLP = mP(learning_rate, num_neurons, len(inputs))

# 初始化权值
weights_in2hid, weights_hid2out = MLP.initial_weights()

# 记录损失
cost_list = []

for i in range(epochs):
    # 前向传播
    a_hidden, z_act_hidden, a_output, z_act_output, x = MLP.forward_pass(
        inputs,
        weights_in2hid,
        weights_hid2out
    )
    # 反向传播
    delta_hid2out, weights_in2hid, weights_hid2out = MLP.back_propagation(
        weights_in2hid,
        weights_hid2out,
        a_hidden,
        x,
        z_act_hidden,
        z_act_output,
        learning_rate,
        targets
    )

    cost = np.mean(np.abs(delta_hid2out))
    cost_list.append(cost)

    if i % 1000 == 0:
        print(f"Iteration:{i}. Error: {cost}")

plt.plot(cost_list)
plt.show()
