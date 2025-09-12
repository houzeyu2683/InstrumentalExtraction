# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

VideoJEPA is a video understanding framework implementing Joint Embedding Predictive Architecture for self-supervised video representation learning. The project focuses on spatiotemporal video analysis with advanced position encoding and transformer-based architectures.

## Core Architecture

### Module Structure

The project is organized into two main modules:

- **`architecture/`**: Neural network models and video processing components
  - `_hamster_.py`: Core VideoJEPA model implementation with encoder-decoder architecture
    - `Hamster` class: Main model with CNN encoder, transformer attention, and decoder
    - `Position` class: Sinusoidal position encoding for sequences
    - Uses encoder-decoder with skip connections for video reconstruction
  - `trainable_position_encoding.py`: Advanced position encoding implementations
    - `TrainablePositionEncoding`: Flexible position encoding (learned/sinusoidal/hybrid)
    - `AdaptivePositionEncoding`: Variable sequence length handling
    - Supports separate temporal-spatial encoding
  - `video_position_encoding.py`: Specialized video position encoding
    - `VideoPositionEncoder`: 3D spatiotemporal position encoding
    - `LoRACompatiblePositionEncoding`: Efficient LoRA-style position encoding
    - Supports patch-based spatial and frame-based temporal encoding

- **`cloud/`**: Data management and video loading infrastructure
  - `_hub_.py`: Video dataset management
    - `Hub` class: Handles train/validation/test video data loading
    - `Library` class: Dataset wrapper with collation functions
    - Supports multiple video formats (.mp4, .avi, .mkv, .mov, .flv, .wmv)
    - Automatic video decoding with torchcodec and torchvision transforms

### Data Structure

```
resource/
├── data/           # Training videos
├── test/          # Test videos  
└── validation/    # Validation videos

Each split contains:
- *.csv files with annotations
- *.mkv video files organized in subdirectories
```

### Model Architecture Details

#### Hamster Model (`_hamster_.py`)
- **Encoder**: Multi-stage 2D CNN (3→4→8→16→32→64→128→256 channels)
- **Position Encoding**: Sinusoidal encoding for sequence positions
- **Attention**: 2-layer Transformer encoder with 8 heads
- **Decoder**: Transpose convolution with skip connections
- **Input**: Videos as (batch, sequence, channels, height, width)
- **Output**: Reconstructed videos with L1 loss

#### Position Encoding Features
- **3D Spatiotemporal**: Full 3D learned/factorized/hierarchical encoding
- **Patch-based**: 16×16 patch encoding for 224×224 images (14×14 patches)
- **Temporal**: Frame-level position encoding up to 64 frames
- **LoRA Compatible**: Low-rank adaptation for efficient fine-tuning

### Video Processing Pipeline

1. **Data Loading**: Hub loads videos from organized directory structure
2. **Preprocessing**: Resize to 224×224, convert to tensors, batch padding
3. **Position Encoding**: Add spatiotemporal position information
4. **Model Forward**: Encoder → Position → Attention → Decoder
5. **Loss Calculation**: L1 reconstruction loss between input and output

## Development Commands

### Environment Setup
```bash
# Install dependencies (PyTorch, torchvision, torchcodec)
pip install torch torchvision torchcodec

# Test data loading
python -c "from cloud import Hub; hub = Hub('./resource'); hub.initiateLibrary()"

# Test model forward pass
python -c "from architecture import Hamster; model = Hamster(512, 'cpu'); model.initiateLayer()"
```

### Key Configuration
- **Video Input**: (batch, frames, 3, height, width)
- **Patch Size**: 16×16 pixels
- **Image Size**: 224×224 pixels  
- **Max Frames**: 64 frames per video
- **Embedding Dimension**: 512 (configurable)
- **Attention Heads**: 8
- **Transformer Layers**: 2

## Implementation Details

### Data Loading (`cloud/_hub_.py`)
- **Video Formats**: Supports .mp4, .avi, .mkv, .mov, .flv, .wmv
- **Batch Collation**: Automatic padding for variable-length videos
- **Transform Pipeline**: PIL resize → tensor conversion → normalization
- **Memory Efficient**: Loads videos on-demand during training

### Position Encoding Types
- **Learned 3D**: Full spatiotemporal parameter matrix
- **Factorized**: Separate temporal + spatial learned encodings  
- **Hierarchical**: Frame-level + 2D patch-level encoding
- **Sinusoidal**: Fixed mathematical position encoding
- **Hybrid**: Learned temporal + sinusoidal spatial

### Model Training Features
- **Skip Connections**: Encoder features reused in decoder
- **Attention Masking**: Supports variable-length sequences
- **Device Agnostic**: Automatic GPU/CPU detection
- **Memory Efficient**: Gradient checkpointing compatible

## Common Tasks

### Adding New Position Encoders
- Extend `TrainablePositionEncoding` with new encoding types
- Implement in `video_position_encoding.py` for video-specific variants
- Follow existing factory pattern for configuration

### Custom Video Processing
- Modify `getCollation` function in `cloud/_hub_.py`
- Add new video transforms in preprocessing pipeline
- Update `Library` class for custom dataset formats

### Model Architecture Changes
- Extend `Hamster` class in `architecture/_hamster_.py`
- Modify encoder/decoder layers in `initiateLayer()` method
- Update loss calculation in `getLoss()` method

### Training Integration
- The project currently has infrastructure but needs training loop
- Data loading and model architecture are complete
- Position encoding systems support various video configurations