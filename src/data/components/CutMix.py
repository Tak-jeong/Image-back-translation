import torch
import numpy as np

def apply_cutmix(images, labels, beta, prob):
    """
    Apply CutMix augmentation.

    Parameters:
    - images (Tensor): Batch of images
    - labels (Tensor): Batch of labels
    - beta (float): Beta distribution parameter for CutMix
    - prob (float): Probability of applying CutMix

    Returns:
    - aug_images (Tensor): Augmented images
    - aug_labels (Tensor): Augmented labels
    """

    def get_rand_bbox(W, H, lam):
        """
        Get random bounding box for CutMix

        Parameters:
        - W (int): Width of the image
        - H (int): Height of the image
        - lam (float): Lambda value sampled from beta distribution

        Returns:
        - (int, int, int, int): Bounding box coordinates (x1, y1, x2, y2)
        """
        cut_rat = np.sqrt(1. - lam)
        cut_w = np.int32(W * cut_rat)
        cut_h = np.int32(H * cut_rat)

        # Uniform
        cx = np.random.randint(W)
        cy = np.random.randint(H)

        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)

        return bbx1, bby1, bbx2, bby2

    # Checking the probability and applying CutMix augmentation
    if torch.rand(1).item() < prob:
        # Generating lambda value from beta distribution
        lam = np.random.beta(beta, beta)

        # Getting the dimensions of the image
        batch_size, channels, H, W = images.shape

        # Getting random bounding box
        bbx1, bby1, bbx2, bby2 = get_rand_bbox(W, H, lam)

        # Getting random permutations of the batch
        rand_perm = torch.randperm(batch_size)

        # Creating the mask for the bounding box
        mask = torch.zeros((batch_size, channels, H, W), device=images.device)
        mask[:, :, bbx1:bbx2, bby1:bby2] = 1

        # Applying CutMix augmentation
        aug_images = images * (1 - mask) + images[rand_perm] * mask
        aug_labels = labels * lam + labels[rand_perm] * (1 - lam)
    else:
        # If not applying CutMix, returning the original images and labels
        aug_images = images
        aug_labels = labels

    return aug_images, aug_labels