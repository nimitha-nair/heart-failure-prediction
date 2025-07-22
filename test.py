import pickle
import numpy as np

model = pickle.load(open("model.pkl", "rb"))
sample = np.array([[50, 1, 200, 0, 35, 1, 250000, 1.5, 137, 1, 0, 4]])
pred = model.predict(sample)
print(pred)