import torch
import unittest

from broadcast import pairwise_similarity


class TestResult(unittest.TestCase):
    def setUp(self):
        self.input = torch.ones(3, 20)
        self.input[1, :] = -1
        self.input[2, :] = 0

    def test_cos(self):
        result = torch.tensor([[ 1., -1.,  0.],
                               [-1.,  1.,  0.],
                               [ 0.,  0.,  0.]])
        self.assertTrue(torch.allclose(pairwise_similarity(self.input, 'cos'), result))

    def test_l1(self):
        result = torch.tensor([[ 0., 40., 20.],
                               [40.,  0., 20.],
                               [20., 20.,  0.]])
        self.assertTrue(torch.allclose(pairwise_similarity(self.input, 'l1'), result))

    def test_l2(self):
        result = torch.tensor([[0.0000, 8.9443, 4.4721],
                               [8.9443, 0.0000, 4.4721],
                               [4.4721, 4.4721, 0.0000]])
        self.assertTrue(torch.allclose(pairwise_similarity(self.input, 'l2'), result))

    def test_error(self):
        with self.assertRaises(ValueError):
            pairwise_similarity(self.input, 'cosine')


if __name__ == '__main__':
    unittest.main()
