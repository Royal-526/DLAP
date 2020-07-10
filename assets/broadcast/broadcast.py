import torch


def naive_pairwise_similarity(input_x: torch.Tensor, method: str) -> torch.Tensor:
    '''
    Calculate similarity matrix with 3 different methods cos, l1, and l2.
    This is a naive implementation with for loop.

    Input
    =====
    input_x: A Tensor with the shape in [batch_size, feature_size]
    method: A string indicate the similarity metric method

    Output
    ======
    similarity_matrix: A Tensor with the shape in [batch_size, batch_size]
    '''

    method_list = ('cos', 'l1', 'l2')
    if method not in method_list: 
        raise ValueError("only support method in {}".format(method_list))
    eps = 1e-23

    batch_size = input_x.size(0)
    similarity_matrix = torch.zeros(batch_size, batch_size)
    for i in range(batch_size):
        for j in range(batch_size):
            if method == 'cos':
                x = input_x[i]
                y = input_x[j]
                inner_product = (x * y).sum()
                x_norm = x.square().sum().sqrt()
                y_norm = y.square().sum().sqrt()
                norm_product = x_norm * y_norm
                similarity_matrix[i, j] = inner_product / (norm_product + eps)
            elif method == 'l1':
                sub = input_x[i] - input_x[j]
                similarity_matrix[i, j] = sub.abs().sum()
            elif method == 'l2':
                sub = input_x[i] - input_x[j]
                similarity_matrix[i, j] = sub.square().sum().sqrt()

    return similarity_matrix


def pairwise_similarity(input_x: torch.Tensor, method: str) -> torch.Tensor:
    '''
    Calculate similarity matrix with 3 different methods cos, l1, and l2

    Input
    =====
    input_x: A Tensor with the shape in [batch_size, feature_size]
    method: A string indicate the similarity metric method

    Output
    ======
    similarity_matrix: A Tensor with the shape in [batch_size, batch_size]
    '''
    method_list = ('cos', 'l1', 'l2')
    if method not in method_list: 
        raise ValueError(f'only support method in {method_list}')
    # TODO: Finish your code here
    batch_size = input_x.size(0)
    similarity_matrix = torch.zeros(batch_size, batch_size)
    if method == 'l1':
        similarity_matrix = ((input_x.unsqueeze(0) - input_x.unsqueeze(1)).abs()).sum(-1)
    elif method == 'l2':
        similarity_matrix = torch.sqrt(((input_x.unsqueeze(0) - input_x.unsqueeze(2))**2).sum(-1))
    elif method == 'cos':
        similarity_matrix = (input_x.unsqueeze(0)*input_x.unsqueeze(1))/(input_x.unsqueeze(0)*input_x.unsqueeze(0))

    return similarity_matrix
    