# dawkins
apply evolution strategies to classification problems.

## why evolution strategies (es)?
es is an optimization technique that learns parameters without backpropagation.

## example
if you know scikit, you know the drill.


```python
# iris dataset
iris = learn.datasets.load_dataset('iris')
X = iris.data
y = np.eye(3)[iris.target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

# create and fit es model
from Dawkins import Dawkins
d = Dawkins(n_pop=200, n_generations=2000)
d.fit(X_train, y_train)
d.predict(X_test, y_test)
```