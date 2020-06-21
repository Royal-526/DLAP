# Broadcast

## Exercises

```
Calculate similarity matrix with 3 different methods cos, l1, and l2

Input
=====
input_x: A Tensor with the shape in [batch_size, feature_size]
method: A string indicate the similarity metric method, including cos, l1, and l2.

Output
======
similarity_matrix: A Tensor with the shape in [batch_size, batch_size]
```

!!! tip
    Try to use the broadcast mechanism instead of using `for` loop to implement it.