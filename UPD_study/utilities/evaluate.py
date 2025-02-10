from argparse import Namespace
from pathlib import Path
from typing import Callable

import torch
import torchvision
import json
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm

from UPD_study.utilities.utils import metrics, log


def get_result_files(eval_dir: Path) -> tuple[Path, Path, Path]:
    return eval_dir / 'anomaly_maps.pt', eval_dir / 'anomaly_scores.pt', eval_dir / 'restored_imgs.pt'


def evaluate(config: Namespace, test_loader: DataLoader, val_step: Callable) -> None:
    """
    Common evaluation method. Handles inference on evaluation set, metric calculation,
    logging and the speed benchmark.

    Args:
        config (Namespace): configuration object.
        test_loader (DataLoader): evaluation set dataloader
        val_step (Callable): validation step function
    """
    print("########Start Evaluation#########")
    # Prepare directories for saving images with masks
    samples_dir = Path(config.eval_dir) / f"sample{config.fold}"
    samples_dir.mkdir(parents=True, exist_ok=True)

    all_metrics = []
    labels = []

    def calculate_metrics(original_img, mask, recon_img, postive):
        """Calculate metrics for a single image."""
        if postive:
            iou = eval_IOU(original_img, recon_img, mask)
            psnr = eval_psnr(original_img, recon_img)
            ssim = eval_ssim(original_img, recon_img)
            dic = eval_dice(original_img, recon_img, mask)  # Assuming DIC is a custom metric
            print(f"IOU: {iou}")
            print(f"dic: {dic}")
            print(f"psnr: {psnr}")
            print(f"ssim: {ssim}")
            return {'iou': iou, 'psnr': psnr, 'ssim': ssim, 'dic': dic}
        else:
            psnr = eval_psnr(original_img, recon_img)
            ssim = eval_ssim(original_img, recon_img)
            print(f"psnr: {psnr}")
            print(f"ssim: {ssim}")
            return {'iou': None, 'psnr': psnr, 'ssim': ssim, 'dic': None}

    def save_sample(idx, original_img, mask, recon_img, diff, filename):
        """Save samples with non-zero masks to disk."""
        sample_dir = samples_dir / f'{idx}_{filename}'
        sample_dir.mkdir(exist_ok=True)
        torchvision.utils.save_image(original_img, sample_dir / 'original.png')
        torchvision.utils.save_image(mask, sample_dir / 'mask.png')
        torchvision.utils.save_image(recon_img, sample_dir / 'reconstructed.png')
        torchvision.utils.save_image(diff, sample_dir / 'diff.png')

    # Iterate over each batch in the test_loader
    for batch_idx, (input_imgs, mask, filepaths) in enumerate(tqdm(test_loader, desc="Evaluating")):
        print(f"Batch {batch_idx}")
        if not config.using_accelerate:
            input_imgs = input_imgs.to(config.device)

        recon_imgs = val_step(input_imgs, test_samples=True).cpu()

        for idx, (original_img, msk, recon_img, filepath) in enumerate(zip(input_imgs, mask, recon_imgs, filepaths)):
            print(idx)
            label = torch.any(msk > 0).item()
            labels.append(label)
            filepath = Path(filepath)
            filename = filepath.name
            diff = get_pred_mask(original_img, recon_img)
            
            print(f"{filename}")

            metrics = calculate_metrics(original_img.cpu(), msk.cpu(), recon_img, label)
            all_metrics.append({'idx': idx, 'filename': filename, **metrics})
            save_sample(batch_idx * len(input_imgs) + idx, original_img.cpu(), msk.cpu(), recon_img, diff, filename)


    # Save metrics to file
    metrics_file = samples_dir / 'metrics.json'
    with open(metrics_file, 'w') as f:
        json.dump(all_metrics, f, indent=4)

# Placeholder functions for computing metrics

def get_pred_mask(origin: torch.Tensor, recon: torch.Tensor) -> torch.Tensor:
    """
    Generates a binary segmentation mask by comparing differences between the original (`origin`) and reconstructed (`recon`) images.
    Typically thresholds pixel-wise reconstruction errors to identify anomalous regions.

    Notes:
        - Ensure `origin` and `recon` have the same shape.
        - The output is a boolean tensor where `True` indicates potential anomalies/regions of interest.
    """
    # Compute absolute difference between original and reconstructed images
    diff = torch.abs(origin - recon)
    
    # Compute mean and std of the difference
    mean = torch.mean(diff)
    std = torch.std(diff)
    
    # Define threshold as mean + 2*std (can be adjusted based on needs)
    threshold = mean + 2 * std
    
    # Generate binary mask where True indicates potential anomalies
    mask = diff > threshold
    
    return mask

def gaussian_kernel(size: int, sigma: float, device: torch.device) -> torch.Tensor:
    """
    Generate a 2D Gaussian kernel for SSIM computation.
    """
    coords = torch.arange(size, device=device).float() - (size - 1) / 2
    coords = coords.view(-1, 1).repeat(1, size)
    
    # Create 2D Gaussian kernel
    gauss = torch.exp(-(coords ** 2 + coords.t() ** 2) / (2 * sigma ** 2))
    
    # Normalize
    return gauss / gauss.sum()

def eval_ssim(origin: torch.Tensor, recon: torch.Tensor) -> torch.Tensor:
    """
    Computes the Structural Similarity Index (SSIM) between `origin` and `recon`, measuring perceptual similarity.

    Notes:
        - Inputs should be normalized to `[0, 1]` and have identical spatial dimensions.
        - Higher values (max 1.0) indicate better reconstruction quality.
    """
    if not (0 <= origin.min() and origin.max() <= 1 and 0 <= recon.min() and recon.max() <= 1):
        raise ValueError("Inputs must be normalized to [0, 1] range")
        
    # SSIM parameters
    K1 = 0.01
    K2 = 0.03
    L = 1.0  # Dynamic range for normalized images
    
    C1 = (K1 * L) ** 2
    C2 = (K2 * L) ** 2
    
    # Generate Gaussian kernel (11x11 is standard, sigma=1.5)
    kernel = gaussian_kernel(11, 1.5, origin.device)
    kernel = kernel.view(1, 1, 11, 11).repeat(origin.size(1), 1, 1, 1)
    
    # Ensure inputs are 4D (batch, channels, height, width)
    if origin.dim() == 3:
        origin = origin.unsqueeze(0)
        recon = recon.unsqueeze(0)
    
    # Compute means
    pad = 11 // 2
    mu1 = F.conv2d(F.pad(origin, (pad, pad, pad, pad), mode='reflect'), 
                   kernel, groups=origin.size(1))
    mu2 = F.conv2d(F.pad(recon, (pad, pad, pad, pad), mode='reflect'), 
                   kernel, groups=recon.size(1))
    
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    
    # Compute variances and covariance
    sigma1_sq = F.conv2d(F.pad(origin * origin, (pad, pad, pad, pad), mode='reflect'), 
                        kernel, groups=origin.size(1)) - mu1_sq
    sigma2_sq = F.conv2d(F.pad(recon * recon, (pad, pad, pad, pad), mode='reflect'), 
                        kernel, groups=recon.size(1)) - mu2_sq
    sigma12 = F.conv2d(F.pad(origin * recon, (pad, pad, pad, pad), mode='reflect'), 
                      kernel, groups=origin.size(1)) - mu1_mu2
    
    # Compute SSIM
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    
    return ssim_map.mean()

def eval_psnr(origin: torch.Tensor, recon: torch.Tensor) -> torch.Tensor:
    """
    Computes Peak Signal-to-Noise Ratio (PSNR) between `origin` and `recon` using Mean Squared Error (MSE).

    Notes:
        - Assumes pixel values are in `[0, 1]` or `[0, 255]`.
        - Higher values (in dB) indicate better fidelity.
    """
    # Validate input ranges
    if torch.max(origin) > 1.0 or torch.min(origin) < 0.0 or \
       torch.max(recon) > 1.0 or torch.min(recon) < 0.0:
        raise ValueError("Inputs must be normalized to [0, 1] range")
    
    # Compute MSE
    mse = torch.mean((origin - recon) ** 2)
    
    # Calculate PSNR (using max_val = 1.0 for normalized inputs)
    psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))
    
    return psnr

def eval_dice(origin: torch.Tensor, recon: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Computes the Dice coefficient between the predicted anomaly mask (from `get_pred_mask(origin, recon)`) and the ground truth `mask`.

    Notes:
        - All inputs must be boolean tensors.
        - Dice measures overlap; ranges from `0` (no overlap) to `1` (perfect match).
    """
    pred_mask = get_pred_mask(origin, recon)
    
    intersection = torch.sum(pred_mask & mask)
    total = torch.sum(pred_mask) + torch.sum(mask)
    
    dice = 2 * intersection / (total + 1e-8)  # Add small epsilon to avoid division by zero
    
    return dice

def eval_IOU(origin: torch.Tensor, recon: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Computes Intersection-over-Union (IoU) between the predicted anomaly mask (from `origin` and `recon`) and `mask`.

    Notes:
        - Inputs must be boolean.
        - IoU ranges from `0` (no overlap) to `1` (identical masks).
    """
    pred_mask = get_pred_mask(origin, recon)
    
    intersection = torch.sum(pred_mask & mask)
    union = torch.sum(pred_mask | mask)
    
    iou = intersection / (union + 1e-8)  # Add small epsilon to avoid division by zero
    
    return iou


def evaluate2(config: Namespace, test_loader: DataLoader, val_step: Callable) -> None:
    """
    Common evaluation method. Handles inference on evaluation set, metric calculation,
    logging and the speed benchmark.

    Args:
        config (Namespace): configuration object.
        test_loader (DataLoader): evaluation set dataloader
        val_step (Callable): validation step function
    """

    labels = []
    anomaly_maps = []
    inputs = []
    segmentations = []
    restored_imgs = []

    # forward pass the testloader to extract anomaly maps, scores, masks, labels
    for input_imgs, mask in tqdm(test_loader, desc="Test set", disable=config.speed_benchmark):
        if not config.using_accelerate:
            input_imgs = input_imgs.to(config.device)

        recon_imgs = val_step(input_imgs, test_samples=True)

        recon_imgs = recon_imgs.cpu() # B, C, H, W
        inputs.append(batch_input_imgs)

        if config.method == 'Cutpaste' and config.localization:
            anomaly_maps.append(batch_anomaly_maps)
            segmentations.append(batch_masks)

            label = torch.where(batch_masks.sum(dim=(1, 2, 3)) > 0, 1, 0)
            labels.append(label)
            anomaly_scores.append(torch.zeros_like(label))
            restored_imgs.append(batch_restored_imgs)
        elif config.method == 'Cutpaste' and not config.localization:
            segmentations = None
            anomaly_scores.append(batch_anomaly_scores)
            label = torch.where(batch_masks.sum(dim=(1, 2, 3)) > 0, 1, 0)
            labels.append(label)
        else:
            anomaly_maps.append(batch_anomaly_maps)
            segmentations.append(batch_masks)
            anomaly_scores.append(batch_anomaly_scores)
            label = torch.where(batch_masks.sum(dim=(1, 2, 3)) > 0, 1, 0)
            labels.append(label)
            restored_imgs.append(batch_restored_imgs)


    metric_prefix = ''
    if config.modality == 'MRI' and config.sequence == 't1':
        metric_prefix = ('brats' if config.brats_t1 else 'atlas') + '/'
    
    metrics(config, anomaly_maps, segmentations, anomaly_scores, labels, restored_imgs, inputs, metric_prefix)

    if config.eval_dir is not None:
        config.eval_dir.mkdir(parents=True, exist_ok=True)
        anomaly_maps_file, anomaly_scores_file, restored_imgs_file = get_result_files(config.eval_dir)
        torch.save(torch.cat(anomaly_maps), anomaly_maps_file)
        torch.save(torch.cat(anomaly_scores), anomaly_scores_file)
        torch.save(torch.cat(restored_imgs), restored_imgs_file)
        # For debugging purposes, save the labels to check ordering is the same
        torch.save(torch.cat(labels), anomaly_scores_file.with_stem('labels'))

    # do a single forward pass to extract images to log
    # the batch size is num_images_log for test_loaders, so only a single forward pass necessary
    input_imgs, mask = next(iter(test_loader))

    if not config.using_accelerate:
        input_imgs = input_imgs.to(config.device)

    output = val_step(input_imgs, test_samples=True)
    if config.using_accelerate:
        input_imgs, mask, output = config.accelerator.gather_for_metrics((input_imgs, mask, output))

    if not config.using_accelerate or config.accelerator.is_main_process:

        anomaly_maps = output[0]

        log({f'anom_val/{metric_prefix}input images': input_imgs.cpu(),
            f'anom_val/{metric_prefix}targets': mask.cpu(),
            f'anom_val/{metric_prefix}anomaly maps': anomaly_maps.cpu()}, config)

        # if recon based method, len(x)==3 because val_step returns reconstructions.
        # if thats the case log the reconstructions
        if len(output) == 3:
            log({f'anom_val/{metric_prefix}reconstructions': output[2].cpu()}, config)
