import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

class ParticleFilter():
    def __init__(self, data, particle=1000):
        self.data = data.values
        self.timelength = len(self.data)
        self.n_particle = particle
        self.sigma_2 = 2**(-2)
        self.alpha_2 = 10**(-1)

        # 潜在変数
        self.x = np.zeros((self.timelength+1, self.n_particle))
        self.x_resampled = np.zeros((self.timelength+1, self.n_particle))

        # 潜在変数の初期値
        initial_x = np.random.normal(0, 1, size=self.n_particle)
        self.x_resampled[0] = initial_x
        self.x[0] = initial_x

        # 重みの初期化
        self.w        = np.zeros((self.timelength, self.n_particle))
        self.w_normed = np.zeros((self.timelength, self.n_particle))

        self.likelihood = np.zeros(self.timelength)

    def __call__(self):
        for t in tqdm(range(self.timelength)):
            for i in range(self.n_particle):
                # Add System noise
                system_noise = np.random.normal(0, np.sqrt(self.alpha_2 * self.sigma_2))
                self.x[t, i] = self.x_resampled[t, i] + system_noise

                # Calculate Likelihood
                self.w[t, i] = self.cal_likelihood(self.x[t,i], self.data[t])

            # Normalize weight W
            self.w_normed[t] = self.w[t]/np.sum(self.w[t])
            self.likelihood[t] = np.log(np.sum(self.w[t]))

            # Resampling
            self.x_resampled[t+1] = self.random_sampling(self.x[t], self.w_normed[t])

        self.draw_graph()

        return 

    def cal_likelihood(self, pred, true):
        return (np.sqrt(2 * np.pi * self.sigma_2))**(-1) * np.exp(-(true - pred)**2 / (2 * self.sigma_2))

    def random_sampling(self, x, prob):
        return np.random.choice(x, size=self.n_particle,  p=prob)

    def get_filtered_value(self):
        return np.diag(np.dot(self.w_normed, self.x[1:].T))

    def draw_graph(self):
        # グラフ描画
        plt.figure(figsize=(16,8))
        plt.plot(range(self.timelength), self.data, label='Grand Truth')
        plt.plot(self.get_filtered_value(), "g", label='Prediction')
        
        for t in range(self.timelength):
            plt.scatter(np.ones(self.n_particle)*t, self.x[t], color="r", s=2, alpha=0.1)
        plt.legend()
        plt.subplots_adjust(left=0.1, right=0.95, bottom=0.1, top=0.95)
        plt.savefig('./Particle_filter.png')
        return