from torch import Tensor


def detach_to_numpy(
    inputs: Tensor,
):
    """Detaches the input tensor and converts it to numpy.

    Args:
        inputs (torch.Tensor): _description_

    Returns:
        ndarray: Converted tensor to ndarray.
    """
    # if input tensor is on gpu, put it on the cpu.
    if inputs.is_cuda:
        return inputs.detach().cpu().numpy()
    # Else just detach and convert to numpy
    return inputs.detach().numpy()
