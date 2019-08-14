
### Questions

### Objectives
YWBAT 

- condition data for a recommender system
- apply cosine similarity to recommend jokes
- describe the pros and cons of using cosine similarity

#### What does cosine similarity measure?
- The angle between two vectors
    - if cosine(v1, v2) == 0 -> perpendicular
    - if cosine(v1, v2) == 1 -> same direction
    - if cosine(v1, v2) == -1 -> opposite direction

### Outline


```python
import pandas as pd
import numpy as np

from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity, cosine_distances


import matplotlib.pyplot as plt
import seaborn as sns
```

### About the data
Format:

- Data files are in .zip format, when unzipped, they are in Excel (.xls) format
- Ratings are real values ranging from -10.00 to +10.00 (the value "99" corresponds to "null" = "not rated").
- One row per user
- The first column gives the number of jokes rated by that user. The next 100 columns give the ratings for jokes 01 - 100.
- The sub-matrix including only columns {5, 7, 8, 13, 15, 16, 17, 18, 19, 20} is dense. Almost all users have rated those jokes (see discussion of "universal queries" in the above paper).



```python
df = pd.read_excel("./data/jester-data-1.xls", header=None)
print(df.shape)
df.head()
```

    (24983, 101)





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>...</th>
      <th>91</th>
      <th>92</th>
      <th>93</th>
      <th>94</th>
      <th>95</th>
      <th>96</th>
      <th>97</th>
      <th>98</th>
      <th>99</th>
      <th>100</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>74</td>
      <td>-7.82</td>
      <td>8.79</td>
      <td>-9.66</td>
      <td>-8.16</td>
      <td>-7.52</td>
      <td>-8.50</td>
      <td>-9.85</td>
      <td>4.17</td>
      <td>-8.98</td>
      <td>...</td>
      <td>2.82</td>
      <td>99.00</td>
      <td>99.00</td>
      <td>99.00</td>
      <td>99.00</td>
      <td>99.00</td>
      <td>-5.63</td>
      <td>99.00</td>
      <td>99.00</td>
      <td>99.00</td>
    </tr>
    <tr>
      <th>1</th>
      <td>100</td>
      <td>4.08</td>
      <td>-0.29</td>
      <td>6.36</td>
      <td>4.37</td>
      <td>-2.38</td>
      <td>-9.66</td>
      <td>-0.73</td>
      <td>-5.34</td>
      <td>8.88</td>
      <td>...</td>
      <td>2.82</td>
      <td>-4.95</td>
      <td>-0.29</td>
      <td>7.86</td>
      <td>-0.19</td>
      <td>-2.14</td>
      <td>3.06</td>
      <td>0.34</td>
      <td>-4.32</td>
      <td>1.07</td>
    </tr>
    <tr>
      <th>2</th>
      <td>49</td>
      <td>99.00</td>
      <td>99.00</td>
      <td>99.00</td>
      <td>99.00</td>
      <td>9.03</td>
      <td>9.27</td>
      <td>9.03</td>
      <td>9.27</td>
      <td>99.00</td>
      <td>...</td>
      <td>99.00</td>
      <td>99.00</td>
      <td>99.00</td>
      <td>9.08</td>
      <td>99.00</td>
      <td>99.00</td>
      <td>99.00</td>
      <td>99.00</td>
      <td>99.00</td>
      <td>99.00</td>
    </tr>
    <tr>
      <th>3</th>
      <td>48</td>
      <td>99.00</td>
      <td>8.35</td>
      <td>99.00</td>
      <td>99.00</td>
      <td>1.80</td>
      <td>8.16</td>
      <td>-2.82</td>
      <td>6.21</td>
      <td>99.00</td>
      <td>...</td>
      <td>99.00</td>
      <td>99.00</td>
      <td>99.00</td>
      <td>0.53</td>
      <td>99.00</td>
      <td>99.00</td>
      <td>99.00</td>
      <td>99.00</td>
      <td>99.00</td>
      <td>99.00</td>
    </tr>
    <tr>
      <th>4</th>
      <td>91</td>
      <td>8.50</td>
      <td>4.61</td>
      <td>-4.17</td>
      <td>-5.39</td>
      <td>1.36</td>
      <td>1.60</td>
      <td>7.04</td>
      <td>4.61</td>
      <td>-0.44</td>
      <td>...</td>
      <td>5.19</td>
      <td>5.58</td>
      <td>4.27</td>
      <td>5.19</td>
      <td>5.73</td>
      <td>1.55</td>
      <td>3.11</td>
      <td>6.55</td>
      <td>1.80</td>
      <td>1.60</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 101 columns</p>
</div>




```python
v1 = np.array([1, 2])
v2 = np.array([1, 2.5])
cosine_similarity(v1.reshape(1, -1), v2.reshape(1, -1)), cosine_distances(v1.reshape(1, -1), v2.reshape(1, -1))
```




    (array([[0.99654576]]), array([[0.00345424]]))



### How do we build a recommender system?
- How do we recommend a joke to userA?
    - user to user ->
        - find users that are similar to userA
        - recommend highly rated jokes that userA has not rated by those users to userA

### Let's condition the data for a recommender system



```python
# we need to replace the 99s with 0s
# but 0 is on the scale...
# moves everything up by 11 and removes the negatives new rating scale is between 1 and 21
# nevermind adding 11 is a terrible idea...

# let's just not do anything...
```


```python
# build a flow for a given user then turn this into a function

user_index = 0
userA = df.drop(0, axis=1).loc[user_index, :]

# let's get the other users
others = df.drop(0, axis=1).drop(index=user_index, axis=0)


# let's find the nearest neighbors
knn = NearestNeighbors(n_neighbors=5, metric='cosine', n_jobs=-1)
knn.fit(others)
```




    NearestNeighbors(algorithm='auto', leaf_size=30, metric='cosine',
                     metric_params=None, n_jobs=-1, n_neighbors=5, p=2, radius=1.0)




```python
distances, indices = knn.kneighbors(userA.values.reshape(1, -1))
distances, indices = distances[0], indices[0]
distances, indices
```




    (array([0.02494242, 0.03028924, 0.0435472 , 0.04501014, 0.04511571]),
     array([22358,  2255,  3509,  5175,  8767]))



#### Now that we have our most similar users, what's next?

#### Find their highest rated items that aren't rated by userA


```python
# let's get jokes not rated by userA
jokes_not_rated = np.where(userA==99)[0]
jokes_not_rated
```




    array([70, 71, 72, 73, 74, 75, 76, 78, 79, 80, 82, 83, 84, 85, 86, 87, 88,
           89, 91, 92, 93, 94, 95, 97, 98, 99])




```python
user_jokes = df.drop(0, axis=1).loc[indices, jokes_not_rated].T.replace(99, 0)
user_jokes['total'] = user_jokes.T.sum()
user_jokes.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>22358</th>
      <th>2255</th>
      <th>3509</th>
      <th>5175</th>
      <th>8767</th>
      <th>total</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>70</th>
      <td>-3.88</td>
      <td>-0.53</td>
      <td>0.0</td>
      <td>-5.97</td>
      <td>4.17</td>
      <td>-6.21</td>
    </tr>
    <tr>
      <th>71</th>
      <td>-9.22</td>
      <td>-4.47</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>-13.69</td>
    </tr>
    <tr>
      <th>72</th>
      <td>-1.17</td>
      <td>7.82</td>
      <td>0.0</td>
      <td>3.20</td>
      <td>0.00</td>
      <td>9.85</td>
    </tr>
    <tr>
      <th>73</th>
      <td>-9.47</td>
      <td>8.83</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>-0.64</td>
    </tr>
    <tr>
      <th>74</th>
      <td>-4.61</td>
      <td>5.92</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>8.83</td>
      <td>10.14</td>
    </tr>
  </tbody>
</table>
</div>




```python
recommend_from = user_jokes['total'].idxmax()
recommend_from
```




    86




```python
# checking our work
user_jokes.ix[86, :] # .loc, .iloc, .ix
```

    /anaconda3/lib/python3.7/site-packages/ipykernel_launcher.py:1: DeprecationWarning: 
    .ix is deprecated. Please use
    .loc for label based indexing or
    .iloc for positional indexing
    
    See the documentation here:
    http://pandas.pydata.org/pandas-docs/stable/indexing.html#ix-indexer-is-deprecated
      """Entry point for launching an IPython kernel.





    22358    8.010
    2255     7.770
    3509     0.000
    5175     0.000
    8767     0.000
    total    3.156
    Name: 86, dtype: float64



### Now let's merge and make a workflow


```python
# build a flow for a given user then turn this into a function
def get_neighbors(userA, others):
    knn = NearestNeighbors(n_neighbors=5, metric='cosine', n_jobs=-1)
    knn.fit(others)
    distances, indices = knn.kneighbors(userA.values.reshape(1, -1))
    distances, indices = distances[0], indices[0]
    return distances, indices


def get_recommended_joke(userA, indices):
    # let's get jokes not rated by userA
    jokes_not_rated = np.where(userA==99)[0]

    user_jokes = df.drop(0, axis=1).loc[indices, jokes_not_rated].T.replace(99, 0)
    user_jokes['total'] = user_jokes.T.sum()

    user_jokes = df.drop(0, axis=1).loc[indices, jokes_not_rated].T.replace(99, 0)
    user_jokes['total'] = user_jokes.T.sum()
    recommended_joke = user_jokes['total'].idxmax()
    return recommended_joke



def recommend_joke(user_index=0):
    userA = df.drop(0, axis=1).loc[user_index, :]
    try:
        # nearest neighbors
        others = df.drop(0, axis=1).drop(index=user_index, axis=0)
        distances, indices = get_neighbors(userA, others)

        # let's get the other users in a dataframe
        recommended_joke = get_recommended_joke(userA, indices)
        return recommended_joke
    except:
        print("user has rated all jokes")
        return None
```


```python
recommend_joke(1923)
```

    user has rated all jokes



```python
df.iloc[1923, :].replace(99, np.nan).isna().sum()
```




    0



### Assessment
- cosine distance
- the recommendation algorithm doesn't always have to use knearestneighbors
- general workflow
- .ix as a slicer for dataframes
- .idxmax to get the index of max value


```python

```
