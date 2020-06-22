import time
import torch
import unittest

from broadcast import pairwise_similarity, naive_pairwise_similarity


class TestResult(unittest.TestCase):
    def setUp(self):
        self.input = torch.ones(3, 20)
        self.input[1, :] = -1
        self.input[2, :] = 0

        self.big_input = torch.randn(200, 20)

    def test_naive_cos(self):
        result = torch.tensor([[ 1., -1.,  0.],
                               [-1.,  1.,  0.],
                               [ 0.,  0.,  0.]])
        self.assertTrue(torch.allclose(naive_pairwise_similarity(self.input, 'cos'), result))

    def test_naive_l1(self):
        result = torch.tensor([[ 0., 40., 20.],
                               [40.,  0., 20.],
                               [20., 20.,  0.]])
        self.assertTrue(torch.allclose(naive_pairwise_similarity(self.input, 'l1'), result))

    def test_naive_l2(self):
        result = torch.tensor([[0.0000, 8.9443, 4.4721],
                               [8.9443, 0.0000, 4.4721],
                               [4.4721, 4.4721, 0.0000]])
        self.assertTrue(torch.allclose(naive_pairwise_similarity(self.input, 'l2'), result))

    def test_naive_error(self):
        with self.assertRaises(ValueError):
            naive_pairwise_similarity(self.input, 'cosine')

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

    def time(self, input_x):
        naive_s = time.time()
        naive_pairwise_similarity(input_x, 'cos')
        naive_pairwise_similarity(input_x, 'l1')
        naive_pairwise_similarity(input_x, 'l2')
        naive_e = time.time()

        broad_s = time.time()
        pairwise_similarity(input_x, 'cos')
        pairwise_similarity(input_x, 'l1')
        pairwise_similarity(input_x, 'l2')
        broad_e = time.time()

        naive_time = naive_e - naive_s
        broad_time = broad_e - broad_s
        return naive_time, broad_time

    def test_small_time(self):
        naive_t, broad_t = self.time(self.input)
        print("\nTest time cost with the input tensor with %s, naive for loop: %.4f, broadcast: %.4f\n" % (self.input.size(), naive_t, broad_t))
        self.assertLess(broad_t, naive_t)

    def test_big_time(self):
        naive_t, broad_t = self.time(self.big_input)
        print("\nTest time cost with the input tensor with %s, naive for loop: %.4f, broadcast: %.4f\n" % (self.big_input.size(), naive_t, broad_t))
        self.assertLess(broad_t, naive_t)


if __name__ == '__main__':
    unittest.main()
