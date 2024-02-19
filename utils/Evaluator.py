import numpy as np

class Evaluator(object):
    def __init__(self):
        self.num_class = 2
        self.confusion_matrix = np.zeros((self.num_class,) * 2)
    
    def add_batch(self, gt_image, pre_image):
        assert gt_image.shape == pre_image.shape
        self.confusion_matrix += self._generate_matrix(gt_image, pre_image)
    
    def _generate_matrix(self, gt_image, pre_image):
        mask = (gt_image >= 0) & (gt_image < self.num_class)
        label = self.num_class * gt_image[mask].astype('int') + pre_image[mask]
        count = np.bincount(label, minlength=self.num_class**2)
        confusion_matrix = count.reshape(self.num_class, self.num_class)
        return confusion_matrix
    
    def reset(self):
        self.confusion_matrix = np.zeros((self.num_class,) * 2)
    
    def dice_coefficient(self):
        """
        Computes the Dice coefficient, a measure of set similarity.
        Returns
        -------
        dice : float
            Dice coefficient as a float on range [0,1].
            Maximum similarity = 1
            No similarity = 0
        """

        intersection = np.diag(self.confusion_matrix)
        union = np.sum(self.confusion_matrix, axis=0) + np.sum(self.confusion_matrix, axis=1) - intersection

        dice = (2 * intersection) / (union+intersection)
        dice = np.nanmean(dice)

        return dice

    def compute_dice(im1, im2, empty_value=1.0):
        """
        Computes the Dice coefficient, a measure of set similarity.
        Parameters
        ----------
        im1 : array-like, bool
            Any array of arbitrary size. If not boolean, will be converted.
        im2 : array-like, bool
            Any other array of identical size as im1. If not boolean, it will be converted.
        empty_value : scalar, float.

        Returns
        -------
        dice : float
            Dice coefficient as a float on range [0,1].
            Maximum similarity = 1
            No similarity = 0
            If both images are empty (sum equal to zero) = empty_value

        Notes
        -----
        The order of inputs for `dice` is irrelevant. The result will be
        identical if `im1` and `im2` are switched.

        This function has been adapted from the Verse Challenge repository:
        https://github.com/anjany/verse/blob/main/utils/eval_utilities.py
        """

        im1 = np.asarray(im1).astype(np.bool_)
        im2 = np.asarray(im2).astype(np.bool_)

        #if im1.shape != im2.shape:
        #    raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

        im_sum = im1.sum() + im2.sum()
        if im_sum == 0:
            return empty_value

        # Compute Dice coefficient
        intersection = np.logical_and(im1, im2)

        return 2.0 * intersection.sum() / im_sum
    