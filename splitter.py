import math
from typing import Dict, List, Tuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from unet import UNet


def batchify(tensor: Tensor, T: int) -> Tensor:
    """
    partition tensor into segments of length T, zero pad any ragged samples
    Args:
        tensor(Tensor): BxCxFxL
    Returns:
        tensor of size (B*[L/T] x C x F x T)
    """
    # Zero pad the original tensor to an even multiple of T
    orig_size = tensor.size(-1)
    new_size = math.ceil(orig_size / T) * T
    tensor = F.pad(tensor, [0, new_size - orig_size])
    # Partition the tensor into multiple samples of length T and stack them into a batch
    return torch.cat(torch.split(tensor, T, dim=-1), dim=0)


class Splitter(nn.Module):
    def __init__(self, stem_names: List[str] = None):
        super(Splitter, self).__init__()

        assert stem_names, "Must provide stem names."
        # stft config
        self.F = 1024
        self.T = 512
        self.win_length = 4096
        self.hop_length = 1024
        self.win = nn.Parameter(torch.hann_window(self.win_length), requires_grad=False)

        self.stems = nn.ModuleDict({name: UNet(in_channels=2) for name in stem_names})

    def compute_stft(self, wav: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Computes stft feature from wav
        Args:
            wav (Tensor): B x L
        """
        stft = torch.stft(
            wav,
            n_fft=self.win_length,
            hop_length=self.hop_length,
            window=self.win,
            center=True,
            return_complex=False,
            pad_mode="constant",
        )

        # only keep freqs smaller than self.F
        stft = stft[:, : self.F, :, :]
        real = stft[:, :, :, 0]
        im = stft[:, :, :, 1]
        mag = torch.sqrt(real**2 + im**2)

        return stft, mag

    def inverse_stft(self, stft: Tensor) -> Tensor:
        """Inverses stft to wave form"""

        pad = self.win_length // 2 + 1 - stft.size(1)
        stft = F.pad(stft, (0, 0, 0, 0, 0, pad))
        wav = torch.istft(
            stft,
            self.win_length,
            hop_length=self.hop_length,
            center=True,
            window=self.win,
        )
        return wav.detach()

    def forward(self, wav: Tensor) -> Dict[str, Tensor]:
        """
        Separates stereo wav into different tracks (1 predicted track per stem)
        Args:
            wav (tensor): 2 x L
        Returns:
            masked stfts by track name
        """
        # stft - 2 X F x L x 2
        # stft_mag - 2 X F x L
        stft, stft_mag = self.compute_stft(wav.squeeze())

        L = stft.size(2)

        # 1 x 2 x F x T
        stft_mag = stft_mag.unsqueeze(-1).permute([3, 0, 1, 2])
        stft_mag = batchify(stft_mag, self.T)  # B x 2 x F x T
        stft_mag = stft_mag.transpose(2, 3)  # B x 2 x T x F

        # compute stems' mask
        masks = {name: net(stft_mag) for name, net in self.stems.items()}

        # compute denominator
        mask_sum = sum([m**2 for m in masks.values()])
        mask_sum += 1e-10

        def apply_mask(mask):
            mask = (mask**2 + 1e-10 / 2) / (mask_sum)
            mask = mask.transpose(2, 3)  # B x 2 X F x T

            mask = torch.cat(torch.split(mask, 1, dim=0), dim=3)

            mask = mask.squeeze(0)[:, :, :L].unsqueeze(-1)  # 2 x F x L x 1
            stft_masked = stft * mask
            return stft_masked

        return {name: apply_mask(m) for name, m in masks.items()}

    def separate(self, wav: Tensor) -> Dict[str, Tensor]:
        """
        Separates stereo wav into different tracks (1 predicted track per stem)
        Args:
            wav (tensor): 2 x L
        Returns:
            wavs by track name
        """

        stft_masks = self.forward(wav)

        return {
            name: self.inverse_stft(stft_masked)
            for name, stft_masked in stft_masks.items()
        }

    @classmethod
    def from_pretrained(cls, model_path: str, from_tensorflow: bool = True):
        if from_tensorflow:
            from tf2pytorch import tf2pytorch

            ckpt = tf2pytorch(checkpoint_path=model_path)
            stem_names = list(
                set([k.split(".")[1] for k in ckpt.keys() if k.startswith("stems.")])
            )
            model = cls(stem_names=stem_names)  # ["vocals", "accompaniment"])
            state_dict = model.state_dict()
            for k, v in ckpt.items():
                if k in state_dict:
                    assert v.shape == state_dict[k].shape
                    state_dict.update({k: torch.from_numpy(v)})
                else:
                    print("Ignore ", k)

            model.load_state_dict(state_dict)
            return model
        raise NotImplementedError("pytorch loading is NOT yet supported.")
