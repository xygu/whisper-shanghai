# Time Series Transformer

A general-purpose time series modeling framework adapted from automatic speech recognition (ASR) architectures. This project demonstrates how core ASR techniques—encoder-decoder transformers, cross-attention mechanisms, and learned positional embeddings—can be effectively migrated to time series forecasting, anomaly detection, and pattern recognition tasks.

## Approach

This model employs a **Transformer sequence-to-sequence architecture** originally developed for speech processing, repurposed for diverse time series tasks including forecasting, imputation, classification, and change point detection. These tasks are jointly represented as a sequence of tokens to be predicted by the decoder, allowing a single model to replace many stages of a traditional time series processing pipeline.

### Key ASR-to-Time-Series Migrations

| ASR Component | Time Series Adaptation |
|:-------------|:----------------------|
| **Log-Mel Spectrogram** | Learnable spectral features from raw time series via 1D convolutions |
| **Encoder-Decoder Architecture** | Encoder processes historical context; decoder generates future predictions |
| **Cross-Attention** | Decoder attends to encoder outputs for long-range temporal dependencies |
| **Positional Embeddings** | Learned or relative positional encodings for irregular time intervals |
| **Token-based Output** | Discretized value tokens for probabilistic multi-step forecasting |

## Setup

Install with pip:

    pip install -U time-series-transformer

Or install from source:

    pip install git+https://github.com/your-org/time-series-transformer.git 

Update to latest:

    pip install --upgrade --no-deps --force-reinstall git+https://github.com/your-org/time-series-transformer.git

## Available Models

| Size | Parameters | Required VRAM | Relative Speed | Best For |
|:---|:---|:---|:---|:---|
| tiny | 39M | ~1 GB | ~10x | Edge devices, real-time streaming |
| base | 74M | ~1 GB | ~7x | Rapid prototyping |
| small | 244M | ~2 GB | ~4x | Balanced accuracy and efficiency |
| medium | 769M | ~5 GB | ~2x | Production forecasting |
| large | 1550M | ~10 GB | 1x | Maximum accuracy |
| turbo | 809M | ~6 GB | ~8x | Fast inference with minimal accuracy loss |

## Command-Line Usage

Forecasting:

    ts-transformer forecast data.csv --model turbo

Anomaly detection:

    ts-transformer detect data.csv --model medium --task anomaly

Imputation:

    ts-transformer impute data.csv --model medium --task impute

View all options:

    ts-transformer --help

## Python Usage

Basic forecasting:

```python
import ts_transformer

model = ts_transformer.load_model("turbo")
result = model.forecast("data.csv")
print(result["predictions"])
```

Lower-level API:

```python
import ts_transformer

model = ts_transformer.load_model("turbo")

# Load and preprocess
ts = ts_transformer.load_series("data.csv")
ts = ts_transformer.normalize(ts)

# Compute spectral features
features = ts_transformer.spectral_features(ts, n_features=model.dims.n_features).to(model.device)

# Decode predictions
options = ts_transformer.DecodingOptions()
result = ts_transformer.decode(model, features, options)
print(result.predictions)
```

## Architectural Details

**Encoder**: Stacked transformer blocks with multi-head self-attention over input sequences. Uses 1D convolutions for local pattern extraction and causal masking to prevent future leakage.

**Decoder**: Autoregressive token generation with cross-attention to encoder outputs for leveraging full historical context.

**Positional Encodings**: Supports sinusoidal, learned, relative, and timestamp-based encodings to handle regular and irregular sampling.

## License

MIT License