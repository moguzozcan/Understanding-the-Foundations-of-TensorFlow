from tensorflow.contrib.learn import datasets

boston = datasets.load_boston()

print(boston.target.shape)