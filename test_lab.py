"""Integration checks against the trained model, not visual-quality claims."""
import unittest
from lab import infer


class ModelIntegration(unittest.TestCase):
    def test_control_effects(self):
        base = infer(40, .5, -1)
        self.assertEqual(base['delta'], 0)
        self.assertEqual(base['baseline'], base['output'])
        ablated = infer(40, .5, 0)
        self.assertGreater(ablated['delta'], 0)
        self.assertEqual(base['baseline'], ablated['baseline'])
        self.assertNotEqual(infer(40, 0, -1)['output'], infer(40, 1, -1)['output'])
        self.assertNotEqual(base['output'], infer(0, .5, -1)['output'])
        self.assertEqual([f['shape'] for f in base['features']],
                         [[16,32,32], [32,16,16], [64,8,8], [32,16,16], [16,32,32]])

    def test_invalid_controls(self):
        for epoch, weight, channel in [(40, 2, -1), (40, .5, 64), (999, .5, -1)]:
            with self.assertRaises(ValueError):
                infer(epoch, weight, channel)


if __name__ == '__main__':
    unittest.main()
