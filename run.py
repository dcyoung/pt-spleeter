from pathlib import Path

import torch
import typer

app = typer.Typer(pretty_exceptions_show_locals=False)


@app.command()
def model_summary() -> None:
    from unet import UNet

    net = UNet()
    print(net)


@app.command()
def test() -> None:
    from unet import UNet

    batch_size = 5
    n_channels = 2
    x = torch.randn(batch_size, n_channels, 512, 128)
    print(x.shape)
    net = UNet(in_channels=n_channels)
    y = net.forward(x)
    print(y.shape)


@app.command()
def split(
    model_path: str = "models/2stems/model",
    input: str = "data/audio_example.mp3",
    output_dir: str = "output",
    offset: float = 0,
    duration: float = 30,
    write_src: bool = False,
) -> None:
    import librosa
    import soundfile

    from splitter import Splitter

    sr = 44100
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    splitter = Splitter.from_pretrained(model_path).to(device).eval()

    # load wav audio
    fpath_src = Path(input)
    wav, _ = librosa.load(
        fpath_src,
        mono=False,
        res_type="kaiser_fast",
        sr=sr,
        duration=duration,
        offset=offset,
    )
    wav = torch.Tensor(wav).to(device)

    # normalize audio
    # wav_torch = wav / (wav.max() + 1e-8)

    with torch.no_grad():
        stems = splitter.separate(wav)

    if write_src:
        stems["input"] = wav
    for name, stem in stems.items():
        fpath_dst = Path(output_dir) / f"{fpath_src.stem}_{name}.wav"
        print(f"Writing {fpath_dst}")
        fpath_dst.parent.mkdir(exist_ok=True)
        soundfile.write(fpath_dst, stem.cpu().detach().numpy().T, sr, "PCM_16")
        # write_wav(fname, np.asfortranarray(stem.squeeze().numpy()), sr)


if __name__ == "__main__":
    app()
