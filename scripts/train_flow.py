from normflow.flow import train_flow
import pandas as pd

df = pd.read_pickle(r'/media/brandon/Data1/Somitogenesis/Dorado/radial_001/spots_raw.pkl')

loss_hist = train_flow(df, max_iter=20_000, input_shape=(10, 12, 12), batch_size=128, hidden_channels=64)