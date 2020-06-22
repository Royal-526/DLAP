# Broadcast

## Exercises

```
Calculate similarity matrix with 3 different methods cos, l1, and l2

Input
=====
input_x: A Tensor with the shape in [batch_size, feature_size]
method: A string indicates the similarity metric method, including cos, l1, and l2.

Output
======
similarity_matrix: A Tensor with the shape in [batch_size, batch_size]
```

!!! tip
    Try to use the broadcast mechanism instead of using `for` loop to implement it.

### 练习文件
[broadcast](./assets/broadcast.py), [test script](./assets/broadcast_test.py)

### 本地测试
将下载的两个文件置于同一个文件下，切换到文件所在目录与所需的python环境后，完成`broadcast.py`中对应的函数，如下图所示：
![](./assets/test1.jpg)
直接运行`python broadcast_test.py`进行测试，这个脚本会测试计算结果的正确性以及耗时，通过测试后得到的结果如图所示：
![](./assets/test2.jpg)
可以看到使用`for`实现的函数远远不如广播机制来的高效。