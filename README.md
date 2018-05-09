# dawkins
apply evolution strategies (es) to classification problems.

## why es?
es is an optimization technique that learns parameters without backpropagation.

## example
if youre familar with scikit estimators, then you should be good to go.
```python
from Dawkins import Dawkins
d = Dawkins(n_pop=200)
d.fit(X_train, y_train)
d.predict(X_test, y_test)
```