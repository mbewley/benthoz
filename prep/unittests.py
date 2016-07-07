import unittest
import os
import shutil

import pandas as pd
import numpy as np

import prep.patches

DATA_SPLIT_DIR = '../data_splits'
TEST_OUTPUT_DIR = '../test_output'
TEST_INPUT_DIR = '../testing_data'
TEST_IMAGE_NAME = 'PR_20081006_232302_383_LC16'


class PatchExtractorTests(unittest.TestCase):
    def setUp(self):
        try:
            os.mkdir(TEST_OUTPUT_DIR)
        except OSError:
            shutil.rmtree(TEST_OUTPUT_DIR)
            os.mkdir(TEST_OUTPUT_DIR)

    def tearDown(self):
        pass
        # shutil.rmtree(TEST_OUTPUT_DIR)

    def test_get_image(self):
        im = prep.patches.get_image(os.path.join(TEST_INPUT_DIR, TEST_IMAGE_NAME + '.png'))
        self.assertEqual(im.shape, (1024, 1360, 3), 'Loaded image shape was incorrect.')
        return im

    def test_get_patches_from_image(self):
        im = self.test_get_image()
        # im = patches.get_image(os.path.join(TEST_INPUT_DIR, TEST_IMAGE_NAME+'.png'))
        PATCHSIZE = 127
        p1 = (119, 647)
        p2 = (155, 50)
        p3 = (942, 892)
        p4 = (989, 365)
        coords_index = pd.MultiIndex.from_tuples([p1, p2, p3, p4], names=['row', 'col'])
        extracted_patches = prep.patches.get_patches_from_image(im=im, coords_index=coords_index, patch_size=PATCHSIZE)

        # Correct number of patches
        self.assertEqual(len(extracted_patches), len(coords_index), 'Incorrect number of patches pulled')

        # Correct patch shapes
        self.assertEqual(extracted_patches[p1].shape, (127, 127, 3), 'Incorrect patch shape')
        self.assertEqual(extracted_patches[p2].shape, (127, 64 + 50, 3), 'Incorrect patch shape')
        self.assertEqual(extracted_patches[p3].shape, (127, 127, 3), 'Incorrect patch shape')
        self.assertEqual(extracted_patches[p4].shape, (63 + 1024 - 989, 127, 3), 'Incorrect patch shape')

        self.assertTrue(np.all(extracted_patches[p1] == im[p1[0] - 63:p1[0] + 64, p1[1] - 63:p1[1] + 64, :]),
                        'Incorrect patch values')
        self.assertTrue(np.all(extracted_patches[p2] == im[p2[0] - 63:p2[0] + 64, :p2[1] + 64, :]),
                        'Incorrect patch values')
        self.assertTrue(np.all(extracted_patches[p3] == im[p3[0] - 63:p3[0] + 64, p3[1] - 63:p3[1] + 64, :]),
                        'Incorrect patch values')
        self.assertTrue(np.all(extracted_patches[p4] == im[p4[0] - 63:, p4[1] - 63:p4[1] + 64, :]),
                        'Incorrect patch values')

        # Test that the means of the first channel (Blue, given opencv reads in BGR format) have not changed.
        self.assertAlmostEqual(extracted_patches[p1][:, :, 0].mean(), 49.5024490049, places=3, msg="Spot check on channel mean fail")
        self.assertAlmostEqual(extracted_patches[p2][:, :, 0].mean(), 86.4618041166, places=3, msg="Spot check on channel mean fail")
        self.assertAlmostEqual(extracted_patches[p3][:, :, 0].mean(), 45.5314030628, places=3, msg="Spot check on channel mean fail")
        self.assertAlmostEqual(extracted_patches[p4][:, :, 0].mean(), 72.5908725695, places=3, msg="Spot check on channel mean fail")

        # Save test patches with fake labels
        labels = pd.Series([2, 238, 2, 238], index=coords_index, name='label')
        prep.patches.write_patches_as_images(TEST_IMAGE_NAME, patches=extracted_patches, labels=labels,
                                             out_dir=TEST_OUTPUT_DIR)

        # Correct patch contents
        for (r, c) in coords_index:
            saved_patch = prep.patches.get_image(
                os.path.join(TEST_OUTPUT_DIR, str(labels[(r, c)]), '{}_{}_{}.png'.format(TEST_IMAGE_NAME, r, c)))
            self.assertTrue(np.all(extracted_patches[(r, c)] == saved_patch), 'Writing and reading patch mismatch')

    def test_dropping_cropped_patches(self):
        im = self.test_get_image()
        # im = patches.get_image(os.path.join(TEST_INPUT_DIR, TEST_IMAGE_NAME+'.png'))
        PATCHSIZE = 127
        p1 = (119, 647)
        p2 = (155, 50)
        p3 = (942, 892)
        p4 = (989, 365)
        coords_index = pd.MultiIndex.from_tuples([p1, p2, p3, p4], names=['row', 'col'])
        extracted_patches = prep.patches.get_patches_from_image(im=im, coords_index=coords_index, patch_size=PATCHSIZE,
                                                                discard_cropped=True)
        # Correct number of patches
        self.assertEqual(len(extracted_patches), 2, 'Incorrect number of patches pulled')


if __name__ == '__main__':
    unittest.main()
