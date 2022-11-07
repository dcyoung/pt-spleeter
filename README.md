# Spleeter  - Pytorch Implementation

This repository houses a from-scratch pytorch implementation of `Spleeter` - for details see the [original repo](https://github.com/deezer/spleeter), and [paper](https://archives.ismir.net/ismir2019/latebreaking/000036.pdf).

The goal of the network is to predict the vocal and instrumental components of an input song provided as an audio spectrogram. Each stem is extracted by a separate UNet architecture similar to a convolutional autoencoder using strided convolutions and extra skip-connections.

![architecture](docs/architecture.jpg)

## Quickstart

```bash
# Clone the repo
git clone https://github.com/dcyoung/pt-spleeter.git
cd pt-spleeter

# Install dependencies
conda env create -f environment.yml
conda activate spleeter

# Download and extract the pretrained model to ./models directory
wget https://github.com/deezer/spleeter/releases/download/v1.4.0/2stems.tar.gz -O /tmp/2stems.tar.gz
mkdir -p ./models/2stems
tar -xf /tmp/2stems.tar.gz -C ./models/2stems

# Extract isolated vocals and accompaniment tracks from a song
python run.py split \
--input=/path/to/song.mp3 \
--model-path=./models/2stems/model
```

## Training

TBD - for now, leveraging official pretrained weights from Spleeter - see [here](https://github.com/deezer/spleeter/wiki/3.-Models). Conversion of the weights from tensorflow to pytorch is supported here.

## References

- Audio spectrogram handling largely lifted from [spleeter-pytorch](https://github.com/tuan3w/spleeter-pytorch)