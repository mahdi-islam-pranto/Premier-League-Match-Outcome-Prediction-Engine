from src.utils import load_object
import numpy as np
model = load_object('artifacts\model.pkl')
test_arr = load_object('artifact\test.csv')
X_test, y_test = test_arr[:, :-1], test_arr[:, -1].astype(int)
print(model.score(X_test, y_test))