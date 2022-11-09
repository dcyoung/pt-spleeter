import random
from pathlib import Path
from typing import List

import musdb
import torch
from torch.utils.data import Dataset


class MusdbDataset(Dataset):
    def __init__(
        self, root="data/musdb18-wav", is_train: bool = True, targets: List[str] = None
    ) -> None:
        super().__init__()
        root = Path(root)
        assert root.exists(), f"Path does not exist: {root}"
        self.mus = musdb.DB(
            root=root,
            subsets=["train" if is_train else "test"],
            is_wav=True,
        )
        self.targets = [s for s in targets] if targets else ["vocals", "accompaniment"]

    def __len__(self) -> int:
        return len(self.mus)

    def __getitem__(self, index):
        track = self.mus.tracks[index]
        track.chunk_duration = 5.0
        track.chunk_start = random.uniform(0, track.duration - track.chunk_duration)
        x_wav = torch.torch.tensor(track.audio.T, dtype=torch.float32)
        y_target_wavs = {
            name: torch.tensor(track.targets[name].audio.T, dtype=torch.float32)
            for name in self.targets
        }
        # original audio (x) and stems (y == targets)
        return x_wav, y_target_wavs
