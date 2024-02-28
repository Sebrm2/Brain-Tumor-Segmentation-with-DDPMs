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
    
    def dice_coef(truth, prediction, batchwise=False):
        '''
        Computes the Sørensen–Dice coefficient for the input matrices. If batchwise=True, the first dimension of the input
        images is assumed to indicate the batch, and this function will return a coefficient for each sample. i.e., images
        of dimension (4,1,20,20,20) would return 4 coefficients.
        Parameters
        ----------
        prediction : np.array
            Array containing the prediction.
        truth : np.array
            Array containing the ground truth.
        batchwise : bool
            Optional. Indicate whether the computation should be done batchwise, assuming that the first dimension of the
            data is the batch. Default: False.
        Returns
        -------
        float or tuple
            Sørensen–Dice coefficient.
        '''

        # Reshape the input to reduce computation to a dot product
        if(not batchwise):
            prediction = np.reshape(prediction, (1,np.prod(prediction.shape)))
            truth = np.reshape(truth, (1,np.prod(truth.shape)))
        else:
            pred_shape = prediction.shape
            prediction = np.reshape(prediction, (pred_shape[0], np.prod(pred_shape[1:])))
            truth_shape = truth.shape
            truth = np.reshape(truth, (truth_shape[0], np.prod(truth_shape[1:])))

        # Prevent values >1 from inflating the score
        np.clip(truth, 0, 1, out=truth)
        np.clip(prediction, 0, 1, out=prediction)

        # Compute dice coef
        coef_list = []
        for i in range(truth.shape[0]):
            coef_denom = np.sum(prediction[i,...]) + np.sum(truth[i,...])
            if(coef_denom == 0):  # If there are no non-zero labels in either the truth or the prediction
                coef_list.append(1.0)  # "Perfect" score
                continue
            coef = prediction[i:i+1, ...] @ truth[i:i+1, ...].T
            coef = 2*coef / coef_denom
            coef_list.append(float(coef))

        # Return list of coeffs if batchwise, otherwise return float
        if(batchwise):
            return tuple(coef_list)
        else:
            return coef_list[0]
        

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
    