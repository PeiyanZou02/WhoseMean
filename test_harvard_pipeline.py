import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

import harvard_pipeline as hp


class HarvardPipelineIntegration(unittest.TestCase):
    def test_mean_and_training_artifacts(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            data, images, output = root/'data', root/'data'/'images', root/'output'
            images.mkdir(parents=True)
            records = []
            for index in range(12):
                pixels = np.zeros((48, 32, 3), dtype=np.uint8)
                pixels[:, :, 0] = 30 + index * 12
                pixels[8:40, 5:27, 1] = 210 - index * 5
                Image.fromarray(pixels).save(images/f'{index}.jpg')
                records.append({'objectid': index, 'localfile': f'{index}.jpg',
                                'classification': 'Paintings' if index < 6 else 'Prints'})
            (data/'records.jsonl').write_text('\n'.join(json.dumps(r) for r in records)+'\n', encoding='utf-8')
            with patch.multiple(hp, DATA=data, IMAGES=images, OUT=output):
                hp.means(size=32)
                hp.train(epochs=1, size=32)
            report = json.loads((output/'report.json').read_text(encoding='utf-8'))
            self.assertEqual(report['artworks'], 12)
            self.assertTrue(report['model_trained'])
            for name in ['pixel_mean.png', 'pixel_median.png', 'model_input_mean.png',
                         'mean_painting.png', 'model.pt', 'history.json']:
                self.assertTrue((output/name).exists(), name)


if __name__ == '__main__':
    unittest.main()
