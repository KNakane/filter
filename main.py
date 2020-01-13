import os, sys
import argparse
import numpy as np
import pandas as pd
from Filter.particle_filter import ParticleFilter
from Filter.kalman_filter import KalmanFilter

def main(args):
    # データの読み込み
    # ダウンロード先：http://daweb.ism.ac.jp/yosoku/
    df = pd.read_csv("./PF-example-data.txt", header=None)
    df.columns = ["data"]
    model = eval(args.r)(data=df)
    model()
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--r', default='ParticleFilter', choices=['ParticleFilter', 'KalmanFilter'])
    args = parser.parse_args()
    main(args)