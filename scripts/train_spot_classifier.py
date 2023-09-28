import pandas as pd
from normflow.classification import train_classifier


df = pd.read_pickle(r'/home/brandon/Documents/Code/zebrafish-ms2/training_data/training_data.pkl')
loss_hist = train_classifier(df, n_epochs=100, learning_rate=1e-4, batch_size=8)

