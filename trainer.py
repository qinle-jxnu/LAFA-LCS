from utils import *

from skimage.metrics import structural_similarity as ssim
from torch.cuda.amp import autocast, GradScaler
use_half_precision_flag = False
scaler = GradScaler()

def train_new(train_loader, model, criterion, sensing_rate, optimizer, device, orth_weight=1e-4):
    model.train()
    sum_loss = 0
    sum_reconstruction_loss = 0
    sum_orthogonality_loss = 0
    scaler = GradScaler()
    base_model = model.module if hasattr(model, "module") else model

    for inputs, _ in train_loader:
        inputs = inputs[:, 0:1, :, :].to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        reconstruction_loss = criterion(outputs, inputs)

        if orth_weight > 0:
            orthogonality_loss = base_model.orthogonality_loss()
        else:
            orthogonality_loss = reconstruction_loss.new_zeros(())

        loss = reconstruction_loss + orth_weight * orthogonality_loss

        loss.backward()
        optimizer.step()

        sum_loss += loss.item()
        sum_reconstruction_loss += reconstruction_loss.item()
        sum_orthogonality_loss += orthogonality_loss.item()

    num_batches = max(len(train_loader), 1)
    return {
        "total": sum_loss / num_batches,
        "reconstruction": sum_reconstruction_loss / num_batches,
        "orthogonality": sum_orthogonality_loss / num_batches,
        "orthogonality_error": base_model.orthogonality_error(),
    }


def valid_bsds(valid_loader, model, criterion, model_name='csdemo', is_test=False):
    sum_psnr = 0
    sum_ssim = 0
    device = test_device
    model.eval()
    with torch.no_grad():
        for iters, (inputs, _) in enumerate(valid_loader):
            inputs = rgb_to_ycbcr(inputs.to(device))[:, 0, :, :].unsqueeze(1) / 255.
            outputs = model(inputs)
            mse = F.mse_loss(outputs[0], inputs)
            psnr = 10 * log10(1 / mse.item())
            sum_psnr += psnr

            sum_ssim += ssim(outputs[0].squeeze().cpu().numpy(), inputs.squeeze().cpu().numpy(), data_range=1)

    return sum_psnr / len(valid_loader), sum_ssim / len(valid_loader)


def valid_set(valid_loader, model, criterion, model_name='csdemo', is_test=False):
    sum_psnr = 0
    sum_ssim = 0
    device = test_device
    model.eval()
    time_sum = 0
    with torch.no_grad():
        for iters, (inputs, _) in enumerate(valid_loader):
            inputs = rgb_to_ycbcr(inputs.to(device))[:, 0, :, :].unsqueeze(1) / 255.
            outputs = model(inputs)
            mse = F.mse_loss(outputs[0], inputs)
            psnr = 10 * log10(1 / mse.item())
            sum_psnr += psnr
            sum_ssim += ssim(outputs[0].squeeze().cpu().numpy(), inputs.squeeze().cpu().numpy(), data_range=1)
    return sum_psnr / len(valid_loader), sum_ssim / len(valid_loader)
