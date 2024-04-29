import os

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from standarddiffusion.dataset.img_dataset import GuidanceType, ImgDataset
from standarddiffusion.diffusion.diffusion import Diffusion
from standarddiffusion.model.unet import DiffusionUNet
from standarddiffusion.training.argparser import get_arguments


def train():

    args = get_arguments()

    # Handle data loading.
    ROOT = os.path.expanduser("~/")

    data_path = os.path.join(ROOT, args.data_path)

    print(f"Reading data from {data_path}")

    dataset = ImgDataset(
        data_path=data_path,
        img_size=tuple(args.img_size),
        cls_guidance=GuidanceType.REG,
    )
    train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    # Handle model loading.
    device = torch.device("cuda" if torch.cuda.is_available() and args.use_cuda else "cpu")

    model = DiffusionUNet().to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr)

    diffusion = Diffusion(
        args.T_steps,
        args.beta_start,
        args.beta_end,
        img_size=tuple(args.img_size),
        device=device,
    )

    loss_fn = torch.nn.CrossEntropyLoss()

    for epoch in range(args.epochs):

        with tqdm(train_loader, desc=f"Epoch {epoch}") as pbar:

            for batch_idx, img in enumerate(pbar):
                pbar.set_description(f"Epoch {epoch} - Batch {batch_idx}: ")

                optimizer.zero_grad()

                # Put data on GPU if available.
                img = img.to(device)
                t = diffusion.sample_timesteps(img.shape[0]).to(device)

                # Sample from the diffusion process.
                x_t, noise = diffusion.q_sample(img, t)
                # Predict noise.
                predicted_noise = model(x_t, t)
                # Calculate loss.
                loss = loss_fn(predicted_noise, noise)

                loss.backward()
                optimizer.step()

                pbar.set_postfix({"loss": loss.item()})

        x, intermediate_samples = diffusion.p_sample_loop(
            model,
            args.batch_size,
            timesteps_to_save=list(range(0, args.T_steps, 100)),
        )

    print("Training Complete.")


if __name__ == "__main__":
    train()
