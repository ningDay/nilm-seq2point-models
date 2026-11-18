import numpy as np


class DataGenerator:
    """
    数据生成器类
    复用并优化自 Notebook 和 `nilm_data.py` 中的内容
    """

    def __init__(self, train_data, val_data, batch_size, seq_len, width,
                 random_seed=1234, binary_threshold=None):
        self.batch_size = batch_size
        assert seq_len % 2 == 1, "seq_len must be odd"
        self.seq_len = seq_len
        self.width = width
        self.random_state = np.random.RandomState(random_seed)
        self.binary_threshold = binary_threshold

        # 数据
        self.train_x = train_data['x']
        self.train_y = train_data['y']
        self.validate_x = val_data['x']
        self.validate_y = val_data['y']

    def calculate_scalar(self, x, y):
        """计算归一化参数"""
        self.x_min, self.x_max = x.min(), x.max()
        self.y_min, self.y_max = y.min(), y.max()
        self.y_mean, self.y_std = y.mean(), y.std()
        return {'y_mean': self.y_mean, 'y_std': self.y_std}

    def transform(self, x, y=None):
        """归一化"""
        x_norm = (x - self.x_min) / (self.x_max - self.x_min + 1e-8)
        if y is None:
            return x_norm
        y_norm = (y - self.y_min) / (self.y_max - self.y_min + 1e-8)
        return x_norm, y_norm

    def inverse_transform_x(self, x):
        return x * (self.x_max - self.x_min + 1e-8) + self.x_min

    def inverse_transform_y(self, y):
        return y * (self.y_max - self.y_min + 1e-8) + self.y_min

    def generate_train(self):
        """生成训练批次"""
        self.random_state.shuffle(self.train_x)

        x = self.train_x
        y = self.train_y
        seq_len = self.seq_len

        offset = (len(x) - seq_len) // self.width

        for i in range(offset):
            batch_x = np.zeros((self.batch_size, seq_len), dtype=np.float32)
            batch_y = np.zeros((self.batch_size, 1), dtype=np.float32)

            for j in range(self.batch_size):
                for k in range(self.width):
                    s = i * self.batch_size + j * self.width + k
                    t = s + seq_len
                    batch_x[j, :] = x[s:t]
                    batch_y[j, 0] = y[t]

            batch_x, batch_y = self.transform(batch_x, batch_y)

            if self.binary_threshold is not None:
                batch_y = (batch_y > self.binary_threshold).astype(np.float32)

            yield batch_x, batch_y

    def generate_validate(self, data_type='val', max_iteration=None):
        """生成验证批次"""
        if data_type == 'val':
            x = self.validate_x
            y = self.validate_y
        else:
            x = self.train_x
            y = self.train_y

        seq_len = self.seq_len
        offset = len(x) - seq_len

        iteration = 0
        while iteration < (max_iteration or offset):
            i = iteration % offset
            batch_x = np.zeros((self.batch_size, seq_len), dtype=np.float32)
            batch_y = np.zeros((self.batch_size, 1), dtype=np.float32)

            for j in range(self.batch_size):
                s = i + j
                t = s + seq_len
                if t < len(x):
                    batch_x[j, :] = x[s:t]
                    batch_y[j, 0] = y[t]

            batch_x, batch_y = self.transform(batch_x, batch_y)

            if self.binary_threshold is not None:
                batch_y = (batch_y > self.binary_threshold).astype(np.float32)

            yield batch_x, batch_y


class TestDataGenerator:
    """测试数据生成器"""

    def __init__(self, test_data, seq_len, x_min, x_max, y_min, y_max, binary_threshold=None):
        self.test_x = test_data['x']
        self.test_y = test_data['y']
        self.seq_len = seq_len
        self.binary_threshold = binary_threshold
        self.x_min, self.x_max = x_min, x_max
        self.y_min, self.y_max = y_min, y_max

    def transform(self, x):
        """归一化"""
        return (x - self.x_min) / (self.x_max - self.x_min + 1e-8)

    def inverse_transform(self, y):
        """反归一化"""
        return y * (self.y_max - self.y_min + 1e-8) + self.y_min

    def generate_inference(self):
        """生成推理数据"""
        x = self.test_x
        seq_len = self.seq_len

        for i in range(len(x) - seq_len):
            batch_x = x[i:i + seq_len].copy()
            batch_x_norm = self.transform(batch_x)
            yield batch_x_norm.reshape(1, -1).astype(np.float32)
