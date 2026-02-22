# Speech Enhancement System

Research-focused speech enhancement system implementing multiple enhancement techniques including deep learning (Conv-TasNet) and traditional spectral methods.

## ⚠️ PRIVACY AND ETHICS DISCLAIMER

**IMPORTANT**: This project is designed for **research and educational purposes only**. 

- **NOT FOR PRODUCTION USE**: This system should not be used for biometric identification, voice cloning, or any production applications
- **RESEARCH DEMO**: Intended for academic research, education, and demonstration purposes
- **PRIVACY PRESERVING**: No raw audio data is stored or transmitted
- **ETHICAL USE**: Users are responsible for ensuring ethical use of this technology
- **MISUSE PROHIBITED**: Voice cloning, deepfake generation, or any form of audio manipulation for deceptive purposes is strictly prohibited

## Features

- **Multiple Enhancement Methods**: Conv-TasNet, Spectral Subtraction, Wiener Filter
- **Comprehensive Evaluation**: SI-SDR, SDR, SNR, PESQ, STOI metrics
- **Synthetic Dataset**: Automatically generated training data
- **Interactive Demo**: Streamlit-based web interface
- **Modern Architecture**: PyTorch 2.x, type hints, proper documentation
- **Reproducible**: Deterministic seeding, configuration management

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/kryptologyst/Speech-Enhancement-System.git
cd Speech-Enhancement-System

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .
```

### Generate Synthetic Dataset

```bash
python scripts/generate_data.py --data_dir data --n_samples 1000
```

### Train a Model

```bash
# Train Conv-TasNet
python scripts/train.py --model_type conv_tasnet --epochs 100 --batch_size 32

# Train Spectral Subtraction
python scripts/train.py --model_type spectral_subtraction --epochs 50

# Train Wiener Filter
python scripts/train.py --model_type wiener_filter --epochs 50
```

### Run Interactive Demo

```bash
streamlit run demo/streamlit_demo.py
```

## Project Structure

```
speech-enhancement-system/
├── src/
│   ├── models/           # Model implementations
│   │   ├── conv_tasnet.py
│   │   └── spectral_methods.py
│   ├── data/             # Dataset classes
│   │   └── synthetic_dataset.py
│   ├── metrics/          # Evaluation metrics
│   │   └── enhancement.py
│   ├── utils/            # Utility functions
│   │   ├── audio.py
│   │   └── device.py
│   └── ...
├── configs/              # Configuration files
├── scripts/              # Training and evaluation scripts
├── demo/                 # Interactive demos
├── tests/                # Unit tests
├── data/                 # Data directory
├── checkpoints/          # Model checkpoints
├── outputs/              # Training outputs
└── logs/                 # Training logs
```

## Models

### Conv-TasNet
- **Architecture**: Encoder-Separator-Decoder
- **Input**: Time-domain waveform
- **Output**: Enhanced speech and separated sources
- **Parameters**: ~5.9M parameters
- **Training**: End-to-end with MSE loss

### Spectral Subtraction
- **Method**: Traditional frequency-domain enhancement
- **Process**: Noise spectrum estimation and subtraction
- **Parameters**: STFT parameters (n_fft, hop_length, window)
- **Advantages**: Fast, interpretable

### Wiener Filter
- **Method**: Optimal linear filtering
- **Process**: Frequency-domain filtering with SNR estimation
- **Parameters**: STFT parameters
- **Advantages**: Theoretically optimal for Gaussian noise

## Dataset

### Synthetic Dataset
- **Speech**: Multi-harmonic synthetic speech signals
- **Noise**: White, pink, and brown noise
- **SNR Range**: 0-20 dB
- **Duration**: Configurable (default 4 seconds)
- **Sample Rate**: 16 kHz

### Data Generation
```python
from src.data.synthetic_dataset import create_synthetic_dataset

train_dataset, val_dataset, test_dataset = create_synthetic_dataset(
    data_dir="data",
    n_samples_train=1000,
    n_samples_val=200,
    n_samples_test=200,
    snr_range=(0, 20)
)
```

## Evaluation Metrics

### Objective Metrics
- **SI-SDR**: Scale-Invariant Signal-to-Distortion Ratio
- **SDR**: Signal-to-Distortion Ratio  
- **SNR**: Signal-to-Noise Ratio
- **PESQ**: Perceptual Evaluation of Speech Quality
- **STOI**: Short-Time Objective Intelligibility

### Usage
```python
from src.metrics.enhancement import EnhancementMetrics

metrics = EnhancementMetrics()
results = metrics.compute_batch_average(enhanced, clean, noisy)
print(f"SI-SDR: {results['si_sdr']:.2f} dB")
```

## Configuration

### Model Configuration
```yaml
# configs/model/conv_tasnet.yaml
encoder_dim: 256
decoder_dim: 256
n_sources: 2
n_layers: 8
n_filters: 512
```

### Data Configuration
```yaml
# configs/data/synthetic.yaml
sample_rate: 16000
duration: 4.0
n_samples_train: 1000
snr_range: [0, 20]
```

## Training

### Basic Training
```bash
python scripts/train.py \
    --model_type conv_tasnet \
    --epochs 100 \
    --batch_size 32 \
    --lr 1e-3 \
    --data_dir data \
    --output_dir outputs
```

### Advanced Training
```bash
python scripts/train.py \
    --config configs/config.yaml \
    --model_type conv_tasnet \
    --epochs 200 \
    --batch_size 64 \
    --lr 1e-3 \
    --device cuda \
    --resume checkpoints/latest.pth
```

## Evaluation

### Model Evaluation
```bash
python scripts/evaluate.py \
    --model_path checkpoints/best.pth \
    --test_data data/test \
    --output_dir results
```

### Metrics Computation
```python
from src.metrics.enhancement import EnhancementMetrics

metrics = EnhancementMetrics()
results = metrics.compute_all(enhanced, clean, noisy)
```

## Demo

### Streamlit Demo
```bash
streamlit run demo/streamlit_demo.py
```

Features:
- Audio upload and recording
- Real-time enhancement
- Metric visualization
- Audio playback and download

### Gradio Demo (Alternative)
```bash
python demo/gradio_demo.py
```

## Development

### Code Quality
```bash
# Format code
black src/ scripts/ demo/

# Lint code
ruff src/ scripts/ demo/

# Run tests
pytest tests/
```

### Pre-commit Hooks
```bash
pre-commit install
pre-commit run --all-files
```

## Performance

### Model Sizes
- **Conv-TasNet**: ~5.9M parameters, ~23.6 MB
- **Spectral Methods**: No trainable parameters

### Inference Speed
- **Conv-TasNet**: ~0.1x real-time on CPU, ~0.01x on GPU
- **Spectral Methods**: ~0.01x real-time on CPU

### Memory Usage
- **Training**: ~2-4 GB GPU memory (batch_size=32)
- **Inference**: ~100-200 MB GPU memory

## Limitations

- **Synthetic Data**: Trained on synthetic data, may not generalize to real-world scenarios
- **Noise Types**: Limited to white, pink, and brown noise
- **Speech Characteristics**: Synthetic speech may not capture all natural speech variations
- **Real-time**: Not optimized for real-time processing

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

This project is licensed under the MIT License. See LICENSE file for details.

## Citation

If you use this code in your research, please cite:

```bibtex
@software{speech_enhancement_system,
  title={Speech Enhancement System},
  author={Kryptologyst},
  year={2026},
  url={https://github.com/kryptologyst/Speech-Enhancement-System}
}
```

## Acknowledgments

- Conv-TasNet paper: "Conv-TasNet: Surpassing Ideal Time-Frequency Magnitude Masking for Speech Separation"
- PyTorch and torchaudio teams
- Librosa contributors
- Streamlit team

## Support

For questions and support:
- Create an issue on GitHub
- Check the documentation
- Review the code comments

---

**Remember**: This is a research demonstration. Use responsibly and ethically.
# Speech-Enhancement-System
