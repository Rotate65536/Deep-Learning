import numpy as np


# 多层感知机模型
class multilayerPerceptron:
    def __init__(
            self,
            learning_rate,
            num_neurons,
            number_of_inputs,
    ):
        self.learning_rate = learning_rate  # 0.1
        self.num_input_neurons = num_neurons[0]  # 2
        self.num_hidden_neurons = num_neurons[1]  # 5
        self.num_output_neurons = num_neurons[2]  # 1
        self.num_inputs = number_of_inputs  # 4

    # 激活函数
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    # 激活函数求导
    def derivative_sigmoid(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))

    # 初始化参数
    def initial_weights(self):
        # num_input_neurons个输入神经元 + 一个偏置(3, 5)
        weights_in2hid = np.random.randn(self.num_input_neurons + 1, self.num_hidden_neurons)
        # num_hidden_neurons个隐层神经元 + 一个偏置(6, 1)
        weights_hid2out = np.random.randn(self.num_hidden_neurons + 1, 1)
        return weights_in2hid, weights_hid2out

    # 前向传播函数
    def forward_pass(
            self,
            inputs,
            weights_in2hid,
            weights_hid2out,
    ):
        # 初始化输入层偏置，将所有偏置设为1
        bias_input = np.ones((self.num_inputs, 1))

        # 将偏置和输入放在一起作为输入层
        x = np.concatenate((bias_input, inputs), axis=1)
        # print("x", x)
        # print("weights_in2hid", weights_in2hid)
        # print("weights_hid2out", weights_hid2out)

        # 计算隐层输入
        # weights_in2hid(3, 5) matmul x(4, 3)
        a_hidden = np.matmul(x, weights_in2hid)
        # print("a_hidden", a_hidden)

        # 计算隐层激活值
        z_act_hidden = self.sigmoid(a_hidden)
        # print("z_act_hidden", z_act_hidden)

        # 初始化隐层偏置，将所有偏置设置为1
        bias_hidden = np.ones((self.num_inputs, 1))
        # print("bias_hidden", bias_hidden)

        # 将偏置和隐层激活值放在一起作为输出层输入
        z_act_hidden = np.concatenate((bias_hidden, z_act_hidden), axis=1)
        # print("z_hidden", z_act_hidden)

        # 计算输出层激活值
        # weights_hid2out(6, 1).T matmul z_hidden(6, 4) = (1, 4)
        a_output = np.matmul(z_act_hidden, weights_hid2out)
        # print("a_output", a_output)
        z_act_output = self.sigmoid(a_output)
        # print("z_act_output", z_act_output)
        return a_hidden, z_act_hidden, a_output, z_act_output, x

    # 反向传播
    def back_propagation(
            self,
            weights_in2hid,
            weights_hid2out,
            a_hidden,
            z0,
            z_act_hidden,
            z_act_output,
            learning_rate,
            targets
    ):
        # 计算输出层delta
        delta_hid2out = z_act_output - targets  # (1, 4)
        # print("delta_hid2out", delta_hid2out)

        # 更新隐层到输出层的权值
        new_w_hid2out = weights_hid2out - learning_rate * np.matmul(z_act_hidden.T,
                                                                    delta_hid2out) / self.num_inputs  # (6, 1) - (1, 6).T
        # print("new_w_hid2out", new_w_hid2out)

        # 计算隐层delta
        delta_in2hid = delta_hid2out.dot(weights_hid2out[1:, :].T) * self.derivative_sigmoid(a_hidden)
        # print("delta_in2hid", delta_in2hid)

        # 更新输入层到隐层的权值
        new_w_in2hid = weights_in2hid - learning_rate * np.matmul(z0.T, delta_in2hid) / self.num_inputs
        # print("new_w_in2hid", new_w_in2hid)
        return delta_hid2out, new_w_in2hid, new_w_hid2out
