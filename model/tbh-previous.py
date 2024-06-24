from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf

# 辅助常量，用于防止计算中出现除零错误
OVERFLOW_MARGIN = 1e-8

# ------------------- 二值化激活函数 -------------------

def sigmoid_sign(logits, eps):
    """
    {0,1} 的符号函数，使用 (1) Sigmoid 激活函数 (2) 在 Sigmoid 中扰动 eps
    :param logits: 底层输出
    :param eps: 随机采样的 [0,1] 之间的值
    :return: 二值化编码和概率
    """
    # 计算 Sigmoid 激活后的概率
    prob = 1.0 / (1 + tf.exp(-logits))
    # 计算扰动后的二值化编码
    code = (tf.sign(prob - eps) + 1.0) / 2.0
    return code, prob

@tf.custom_gradient
def binary_activation(logits, eps):
    """
    自定义的二值化激活函数，并带有梯度计算。
    :param logits: 输入张量
    :param eps: 随机扰动值
    :return: 二值化编码和概率
    """
    code, prob = sigmoid_sign(logits, eps)

    def grad(_d_code, _d_prob):
        """
        基于 Bernoulli 概率的分布式导数
        :param _d_code: 通过编码的反向传播梯度
        :param _d_prob: 通过概率的反向传播梯度
        :return: logits 和 eps 的梯度
        """
        # 计算 logits 的梯度
        d_logits = prob * (1 - prob) * (_d_code + _d_prob)
        # eps 的梯度直接等于 _d_code
        d_eps = _d_code
        return d_logits, d_eps

    return [code, prob], grad

# ------------------- 图卷积网络层 -------------------

class GCNLayer(tf.keras.layers.Layer):
    def __init__(self, out_dim, **kwargs):
        super().__init__(**kwargs)
        # 输出维度
        self.out_dim = out_dim
        # 全连接层
        self.fc = tf.keras.layers.Dense(out_dim)

    def call(self, values, adjacency, **kwargs):
        """
        在图上应用谱卷积。
        :param values: 输入特征矩阵 [N, D]
        :param adjacency: 邻接矩阵 [N, N]
        :return: 卷积后的特征矩阵
        """
        return self.spectrum_conv(values, adjacency)

    @tf.function
    def spectrum_conv(self, values, adjacency):
        """
        基于图拉普拉斯算子的图卷积。
        :param values: 输入特征矩阵 [N, D]
        :param adjacency: 邻接矩阵 [N, N]
        :return: 卷积后的特征矩阵
        """
        # 通过全连接层计算特征
        fc_sc = self.fc(values)
        # 通过图拉普拉斯算子进行卷积
        conv_sc = self.graph_laplacian(adjacency) @ fc_sc
        return conv_sc

    @staticmethod
    @tf.function
    def graph_laplacian(adjacency):
        """
        计算图拉普拉斯算子。
        :param adjacency: 必须是自连接的邻接矩阵
        :return: 图拉普拉斯矩阵
        """
        graph_size = tf.shape(adjacency)[0]
        # 度矩阵
        d = adjacency @ tf.ones([graph_size, 1])
        # 度矩阵的逆平方根
        d_inv_sqrt = tf.pow(d + OVERFLOW_MARGIN, -0.5)
        d_inv_sqrt = tf.eye(graph_size) * d_inv_sqrt
        # 计算图拉普拉斯矩阵
        laplacian = d_inv_sqrt @ adjacency @ d_inv_sqrt
        return laplacian

# ------------------- 双瓶颈层 -------------------

@tf.function
def build_adjacency_hamming(tensor_in):
    """
    基于汉明距离的图构建，图是自连接的。
    :param tensor_in: 输入张量 [N, D]
    :return: 邻接矩阵
    """
    code_length = tf.cast(tf.shape(tensor_in)[1], tf.float32)
    m1 = tensor_in - 1
    # 计算两个汉明距离矩阵
    c1 = tf.matmul(tensor_in, m1, transpose_b=True)
    c2 = tf.matmul(m1, tensor_in, transpose_b=True)
    # 计算标准化的距离矩阵
    normalized_dist = tf.math.abs(c1 + c2) / code_length
    return tf.pow(1 - normalized_dist, 1.4) #调整超参数

class TwinBottleneck(tf.keras.layers.Layer):
    def __init__(self, bbn_dim, cbn_dim, **kwargs):
        super().__init__(**kwargs)
        self.bbn_dim = bbn_dim
        self.cbn_dim = cbn_dim
        # 图卷积网络层
        self.gcn = GCNLayer(cbn_dim)

    def call(self, bbn, cbn):
        # 构建基于汉明距离的邻接矩阵
        adj = build_adjacency_hamming(bbn)
        # 对连续瓶颈应用图卷积网络并返回结果
        return tf.nn.sigmoid(self.gcn(cbn, adj))

# ------------------- 编码器和解码器 -------------------

class Encoder(tf.keras.layers.Layer):
    def __init__(self, middle_dim, bbn_dim, cbn_dim):
        """
        初始化编码器。
        :param middle_dim: 隐藏层单元数
        :param bbn_dim: 二值瓶颈大小
        :param cbn_dim: 连续瓶颈大小
        """
        super(Encoder, self).__init__()
        self.code_length = bbn_dim
        # 第一个全连接层
        self.fc_1 = tf.keras.layers.Dense(middle_dim, activation='relu')
        # 二值瓶颈全连接层
        self.fc_2_1 = tf.keras.layers.Dense(bbn_dim)
        # 连续瓶颈全连接层
        self.fc_2_2 = tf.keras.layers.Dense(cbn_dim, activation='sigmoid')

    def call(self, inputs, training=True, **kwargs):
        batch_size = tf.shape(inputs)[0]
        # 通过第一个全连接层
        fc_1 = self.fc_1(inputs)
        # 通过二值瓶颈全连接层
        bbn = self.fc_2_1(fc_1)
        # 初始化 eps 为 0.5
        eps = tf.ones([batch_size, self.code_length]) / 2.
        # 计算二值化编码和概率
        bbn, _ = binary_activation(bbn, eps)
        # 通过连续瓶颈全连接层
        cbn = self.fc_2_2(fc_1)
        return bbn, cbn

class Decoder(tf.keras.layers.Layer):
    def __init__(self, middle_dim, feat_dim):
        """
        初始化解码器。
        :param middle_dim: 隐藏层单元数
        :param feat_dim: 数据维度
        """
        super(Decoder, self).__init__()
        # 第一个全连接层
        self.fc_1 = tf.keras.layers.Dense(middle_dim, activation='relu')
        # 第二个全连接层
        self.fc_2 = tf.keras.layers.Dense(feat_dim, activation='relu')

    def call(self, inputs, **kwargs):
        # 通过第一个全连接层
        fc_1 = self.fc_1(inputs)
        # 通过第二个全连接层并返回
        return self.fc_2(fc_1)

# ------------------- TBH 模型 -------------------

class TBH(tf.keras.Model):
    def __init__(self, set_name, bbn_dim, cbn_dim, middle_dim=1024, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.set_name = set_name
        self.bbn_dim = bbn_dim
        self.cbn_dim = cbn_dim
        self.middle_dim = middle_dim
        # 假设特征维度是 4096
        self.feat_dim = 4096  # Assuming SET_DIM.get(set_name, 4096)

        # 初始化编码器、解码器和双瓶颈层
        self.encoder = Encoder(middle_dim, bbn_dim, cbn_dim)
        self.decoder = Decoder(middle_dim, self.feat_dim)
        self.tbn = TwinBottleneck(bbn_dim, cbn_dim)

        # 判别器的全连接层
        self.dis_1 = tf.keras.layers.Dense(1, activation='sigmoid')
        self.dis_2 = tf.keras.layers.Dense(1, activation='sigmoid')

    def call(self, inputs, training=True, mask=None):
        # 获取输入特征
        feat_in = tf.cast(inputs[0][1], dtype=tf.float32)
        # 编码器得到二值瓶颈和连续瓶颈
        bbn, cbn = self.encoder(feat_in, training=training)

        if training:
            # 训练时通过双瓶颈层得到新的表示
            bn = self.tbn(bbn, cbn)
            dis_1 = self.dis_1(bbn)
            dis_2 = self.dis_2(bn)
            feat_out = self.decoder(bn)
            sample_bbn = inputs[1]
            sample_bn = inputs[2]
            dis_1_sample = self.dis_1(sample_bbn)
            dis_2_sample = self.dis_2(sample_bn)
            return bbn, feat_out, dis_1, dis_2, dis_1_sample, dis_2_sample
        else:
            # 推理时只返回二值瓶颈
            return bbn

# ------------------- 测试主函数 -------------------

if __name__ == '__main__':
    # 创建一个输入张量
    a = tf.ones([2, 4096], dtype=tf.float32)
    # 初始化编码器
    encoder = Encoder(1024, 64, 512)
    # 获取编码器的输出
    b = encoder(a)
    # 打印编码器的可训练变量
    print(encoder.trainable_variables)
