import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

class KalmanFilter():
    def __init__(self, data, dim=1):
        self.data = data.values
        self.timelength = len(self.data)

        # 潜在変数
        self.x = np.zeros((self.timelength+1, dim))
        self.x_filter = np.zeros((self.timelength+1, dim))
        # 共分散行列
        self.sigma = np.zeros((self.timelength+1, dim))
        self.sigma_filter = np.zeros((self.timelength+1, dim))

        # 状態遷移行列
        self.A = np.ones(dim)

        # 観測行列
        self.C = np.ones(dim)

        # ノイズ
        self.Q = 1.0
        self.R = 1.0
        self.W = np.random.normal(loc=0, scale=self.Q, size=self.x.shape)
        self.V = np.random.normal(loc=0, scale=self.R, size=self.x.shape)

    def __call__(self):
        #for t in tqdm(range(self.timelength-1)):
        for t in (range(self.timelength-1)):
            # 状態量推定
            self.x[t+1] = self.A * self.x[t] + self.W[t]
            self.sigma[t+1] = self.Q + self.A * self.sigma[t] * self.A.T

            # 更新
            #Kalman_gain = self.sigma[t+1] * self.C.T * (self.C * self.sigma[t+1] * self.sigma[t+1].T + self.R).T
            Kalman_gain = self.sigma[t+1] / (self.sigma[t+1] + self.R)

            self.x_filter[t+1] = self.x[t+1] + Kalman_gain * (self.data[t+1] - self.C * self.x[t+1])
            self.sigma_filter[t+1] = self.sigma[t+1] - Kalman_gain * self.C * self.sigma[t+1]

        self.draw_graph()
        return

    def draw_graph(self):
        # グラフ描画
        plt.figure(figsize=(16,8))
        plt.plot(range(self.timelength), self.data, label='Grand Truth')
        plt.plot(range(self.timelength), self.x_filter[:-1], "g", label='Prediction')
        plt.legend()
        plt.subplots_adjust(left=0.1, right=0.95, bottom=0.1, top=0.95)
        plt.savefig('./Kalman_filter.png')
        return
