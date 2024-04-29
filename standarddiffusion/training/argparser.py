import argparse


def get_arguments():
    parser = argparse.ArgumentParser(description="Process some integers.")

    # Data arguments.
    parser.add_argument(
        "--data_path",
        type=str,
        help="Path for the data directory. The data directory should contain images.",
    )
    parser.add_argument(
        "--img_size",
        type=int,
        nargs=2,
        default=(32, 32),
        help="Size of the images to be resized to. Example: --img_width 32 32",
    )

    # Diffusion arguments.

    # Model arguments.

    # Training arguments.
    parser.add_argument(
        "--use_cuda",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use CUDA for training if available.",
    )
    parser.add_argument(
        "--lr",
        type=int,
        default=1e-3,
        help="Learning rate for the optimizer.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of epochs to train the model for.",
    )

    return parser.parse_args()
