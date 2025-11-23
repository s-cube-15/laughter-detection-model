# Laughter Detection Model 🎭😂

A deep learning-based system for detecting and segmenting laughter in audio recordings using ResNet architectures and advanced audio feature extraction techniques.

## 🎯 Overview

This project implements an end-to-end audio classification pipeline that automatically detects laughter segments in speech recordings. The system uses convolutional neural networks (ResNet) trained on mel-spectrogram features to identify laughter with precise temporal boundaries.

**Key Features:**
- 🧠 ResNet-based deep learning architecture with residual blocks
- 🎵 Advanced audio feature extraction (mel-spectrograms, MFCC)
- 🔄 Data augmentation pipeline (SpecAugment, waveform perturbations)
- 📊 Trained on AudioSet and Switchboard conversational datasets
- ⏱️ Temporal segmentation with configurable sensitivity thresholds
- 📈 Comprehensive evaluation metrics (precision, recall, F1-score)

## 🏗️ Architecture

### Model Configurations

The project supports multiple model architectures:

1. **MLP Baseline (MFCC)** - Multi-layer perceptron using MFCC features
2. **ResNet Base** - ResNet without augmentation for baseline comparison
3. **ResNet with Augmentation** - Enhanced ResNet with SpecAugment and waveform augmentation (recommended)

### Network Architecture

```
Input Audio (8kHz) 
    ↓
Mel-Spectrogram Feature Extraction (hop_length=186, 100 FPS)
    ↓
ResNet Architecture:
  - Conv2D (64 filters, 3x3 kernel)
  - Batch Normalization
  - 4 Residual Blocks [128→64→32→32 filters]
  - Global Average Pooling
  - Fully Connected Layers (Linear → Dropout → Sigmoid)
    ↓
Binary Classification (Laughter / Non-Laughter)
    ↓
Temporal Segmentation (Lowpass Filter → Threshold → Min Duration)
    ↓
Output: [(start_time, end_time), ...]
```

## 📦 Installation

### Prerequisites

- Python 3.7+
- CUDA-capable GPU (optional but recommended)

### Setup

```bash
# Clone the repository
git clone https://github.com/s-cube-15/laughter-detection-model.git
cd laughter-detection-model

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Dependencies

```
torch==1.7.1
librosa==0.8.1
tgt==1.4.4
pyloudnorm==0.1.0
praatio==3.8.0
tensorboardX==1.9
pandas
numpy==1.21.5
scikit-learn==1.0.2
streamlit==1.7.0
```

## 🚀 Usage

### 1. Inference: Detect Laughter in Audio Files

```bash
python segment_laughter.py \
    --input_audio_file=path/to/audio.wav \
    --output_dir=./output \
    --threshold=0.5 \
    --min_length=0.2 \
    --save_to_audio_files=True \
    --save_to_textgrid=False
```

**Parameters:**
- `--input_audio_file`: Path to input audio file (WAV format)
- `--output_dir`: Directory to save output files
- `--threshold`: Detection threshold (0.0-1.0, default: 0.5)
- `--min_length`: Minimum laughter duration in seconds (default: 0.2)
- `--save_to_audio_files`: Extract laughter segments as separate audio files
- `--save_to_textgrid`: Save results in Praat TextGrid format

### 2. Training a Model

```bash
# Train ResNet with augmentation
python train.py \
    --config=resnet_with_augmentation \
    --batch_size=32 \
    --checkpoint_dir=./checkpoints/my_model

# Train on AudioSet (noisy data)
python train.py \
    --config=resnet_with_augmentation \
    --batch_size=32 \
    --checkpoint_dir=./checkpoints/audioset_model \
    --train_on_noisy_audioset=True
```

### 3. Streamlit Web Application

```bash
streamlit run app.py
```

Open your browser and navigate to `http://localhost:8501` to use the interactive web interface.

## 📊 Dataset

The model is trained on two primary datasets:

### 1. **AudioSet**
- Large-scale audio event dataset
- Contains diverse laughter samples from various contexts
- Includes noisy real-world recordings

### 2. **Switchboard Corpus**
- Telephonic conversational speech
- High-quality annotations
- Used for validation and testing

Dataset structure:
```
data/
├── audioset/
│   ├── annotations/
│   ├── splits/
│   └── AI_open_mic_dataset/
└── switchboard/
    ├── train/
    ├── val/
    └── test/
```

## 🎓 Model Training Details

### Feature Extraction
- **Sampling Rate:** 8000 Hz
- **Feature Type:** Mel-spectrograms (default) or MFCC
- **Hop Length:** 186 samples
- **Frame Rate:** 100 FPS

### Data Augmentation
- **SpecAugment:** Time and frequency masking
- **Waveform Augmentation:** Time-stretching, pitch-shifting
- **Loudness Normalization:** Using pyloudnorm

### Training Configuration
- **Optimizer:** SGD with momentum
- **Learning Rate:** 0.01 (decay: 0.9999)
- **Batch Size:** 32
- **Dropout Rate:** 0.5
- **Training Steps:** 100,000
- **Device:** CUDA (GPU) or CPU

## 📈 Evaluation

The project includes comprehensive evaluation scripts:

```bash
# Evaluate on Switchboard test set
python scripts/Evaluation/evaluate_resnet_specaug_wavaug_on_switchboard.py

# Evaluate on AudioSet
python scripts/Evaluation/evaluate_resnet_specaug_wavaug_on_audioset.py

# Analyze results with bootstrap confidence intervals
python scripts/Evaluation/analyze_results.py
```

### Metrics
- **Frame-level Accuracy:** Percentage of correctly classified frames
- **Precision:** True positive rate among predicted laughter
- **Recall:** True positive rate among actual laughter
- **F1-Score:** Harmonic mean of precision and recall
- **Event-based Metrics:** Detection of laughter events (start/end)

## 🗂️ Project Structure

```
laughter-detection-model/
├── app.py                      # Streamlit web application
├── segment_laughter.py         # Inference script
├── train.py                    # Training script
├── model.py                    # Main model notebook
├── models.py                   # Neural network architectures
├── configs.py                  # Model configurations
├── laugh_segmenter.py          # Segmentation utilities
├── rating_humour_quotient.py   # Humor rating module
├── requirements.txt            # Python dependencies
├── checkpoints/
│   ├── in_use/                 # Active model checkpoints
│   └── comparisons/            # Baseline comparisons
├── data/                       # Training/validation data
├── scripts/
│   ├── Evaluation/             # Evaluation scripts
│   ├── download_audioset_*.py  # Data download utilities
│   └── aggregate_*.py          # Annotation processing
└── utils/
    ├── audio_utils.py          # Audio processing functions
    ├── data_loaders.py         # PyTorch data loaders
    ├── dataset_utils.py        # Dataset utilities
    ├── torch_utils.py          # PyTorch helpers
    └── text_utils.py           # Text processing
```

## 🔬 Technical Details

### Temporal Segmentation Algorithm

1. **Frame-level Classification:** Model predicts laughter probability for each audio frame (100 FPS)
2. **Lowpass Filtering:** Butterworth filter smooths probability curve
3. **Thresholding:** Frames above threshold are marked as laughter
4. **Boundary Detection:** Consecutive laughter frames are grouped
5. **Duration Filtering:** Segments shorter than minimum duration are discarded
6. **Time Conversion:** Frame indices converted to seconds

### Preprocessing Pipeline

```python
Audio File (any format)
    ↓
Load with Librosa (resample to 8kHz)
    ↓
Extract Mel-Spectrogram Features
    ↓
Apply Augmentation (training only)
    ↓
Normalize and Pad Sequences
    ↓
Convert to PyTorch Tensor
    ↓
Feed to Model
```

## 🎯 Use Cases

- **Comedy Analysis:** Analyze stand-up comedy performances
- **Meeting Analytics:** Detect engagement in video conferences
- **Content Moderation:** Identify laugh tracks in media
- **Research:** Study social dynamics and humor patterns
- **Accessibility:** Generate laugh captions for hearing-impaired users
- **Entertainment:** Interactive humor rating systems

## 📝 Example Output

```python
[
    {'start': 2.34, 'end': 3.12},
    {'start': 5.67, 'end': 6.89},
    {'start': 12.45, 'end': 14.23}
]
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- AudioSet dataset by Google Research
- Switchboard Corpus by LDC
- PyTorch and Librosa communities
- ResNet architecture inspiration

## 📧 Contact

**Sudhanshu Sabale**
- Email: sudhanshussable2@gmail.com
- GitHub: [@s-cube-15](https://github.com/s-cube-15)
- LinkedIn: [Sudhanshu Sabale](https://www.linkedin.com/in/sudhanshu-sabale-28ab4520a/)

## 🌟 Citation

If you use this project in your research, please cite:

```bibtex
@software{sabale2024laughter,
  author = {Sabale, Sudhanshu},
  title = {Laughter Detection Model},
  year = {2024},
  url = {https://github.com/s-cube-15/laughter-detection-model}
}
```

---

⭐ **Star this repository if you find it helpful!**
