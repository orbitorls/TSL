"""Smoke tests for model forward/backward passes."""

import unittest

import torch
import torch.nn as nn

from src.train.models import MLP, GRUModel


class TrainSmokeTest(unittest.TestCase):
    def test_gru_forward_backward(self):
        model = GRUModel(input_dim=162, num_classes=51, hidden_dim=64, num_layers=2, dropout=0.1)
        x = torch.randn(4, 162)
        y = torch.randint(0, 51, (4,))
        logits = model(x)
        self.assertEqual(logits.shape, (4, 51))
        loss = nn.functional.cross_entropy(logits, y)
        loss.backward()
        for p in model.parameters():
            self.assertIsNotNone(p.grad)

    def test_mlp_forward_backward(self):
        model = MLP(input_dim=162, num_classes=51, hidden_dim=64, num_layers=2, dropout=0.1)
        x = torch.randn(4, 162)
        y = torch.randint(0, 51, (4,))
        logits = model(x)
        self.assertEqual(logits.shape, (4, 51))
        loss = nn.functional.cross_entropy(logits, y)
        loss.backward()
        for p in model.parameters():
            self.assertIsNotNone(p.grad)

    def test_gru_sequence_input(self):
        model = GRUModel(input_dim=162, num_classes=51, hidden_dim=64, num_layers=2)
        x = torch.randn(2, 30, 162)
        logits = model(x)
        self.assertEqual(logits.shape, (2, 51))


if __name__ == '__main__':
    unittest.main()
