"""Verify class filtering against the installed TensorFlow/Keras loader."""
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import train_float32
import train_qat
import export_tflite
from prepare_dataset import read_classes, write_json


class TrainingClassTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        write_json(self.root / 'dataset.json', {'classes': ['LEFT', 'RIGHT', 'STRAIGHT']})
        for split in ('train', 'val', 'test'):
            for label, value in (('LEFT', 20), ('RIGHT', 100), ('STRAIGHT', 200)):
                folder = self.root / split / label
                folder.mkdir(parents=True)
                cv2.imwrite(str(folder / 'one.jpg'), np.full((16, 16, 3), value, np.uint8))
        for module in (train_float32, train_qat, export_tflite):
            self.addCleanup(setattr, module, 'CLASS_NAMES', module.CLASS_NAMES)

    def test_two_class_training_and_qat_loaders_ignore_left_and_remap_labels(self):
        for module in (train_float32, train_qat):
            module.CLASS_NAMES = read_classes(self.root, ['RIGHT', 'STRAIGHT'])
            for split in ('train', 'val', 'test'):
                with self.subTest(module=module.__name__, split=split):
                    batches = list(module.make_dataset(self.root, split, 2, False, 42).as_numpy_iterator())
                    images, labels = batches[0]
                    self.assertEqual(len(batches), 1)
                    np.testing.assert_allclose(images.mean(axis=(1, 2, 3)), [100, 200], atol=1)
                    np.testing.assert_array_equal(labels, [[1, 0], [0, 1]])

    def test_all_classes_still_load_without_filtering(self):
        train_float32.CLASS_NAMES = read_classes(self.root)
        batches = list(train_float32.make_dataset(self.root, 'train', 3, False, 42).as_numpy_iterator())
        np.testing.assert_array_equal(batches[0][1], np.eye(3))

    def test_empty_unselected_left_is_allowed_but_selected_left_is_rejected(self):
        (self.root / 'train/LEFT/one.jpg').unlink()
        train_float32.CLASS_NAMES = ['RIGHT', 'STRAIGHT']
        samples = list(train_float32.make_dataset(self.root, 'train', 2, False, 42).as_numpy_iterator())
        self.assertEqual(samples[0][1].shape, (2, 2))
        train_float32.CLASS_NAMES = ['LEFT', 'RIGHT', 'STRAIGHT']
        with self.assertRaisesRegex(ValueError, 'train/LEFT'):
            train_float32.make_dataset(self.root, 'train', 2, False, 42)

    def test_quantization_calibration_uses_only_selected_classes(self):
        export_tflite.CLASS_NAMES = read_classes(self.root, ['RIGHT', 'STRAIGHT'])
        examples = list(export_tflite.representative_dataset(self.root, 10, 42))
        self.assertEqual(len(examples), 2)
        means = sorted(float(example[0].numpy().mean()) for example in examples)
        np.testing.assert_allclose(means, sorted([100 / 127.5 - 1, 200 / 127.5 - 1]), atol=0.01)


if __name__ == '__main__':
    unittest.main()
