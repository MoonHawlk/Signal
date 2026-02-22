# Signal

Real-Time Circular Chladni Plate Simulation using Python, OpenGL and Live Audio.

---

## Overview

**Signal** captures microphone audio in real time, computes its frequency spectrum,
and feeds it into a GPU shader that simulates a vibrating circular plate.

The result is a real-time Chladni-like pattern that responds dynamically to your voice.

---

## How It Works

### Pipeline

Microphone → FFT → Mode Mapping → OpenGL Shader → Circular Plate → Nodal Lines

### Steps

1. Audio is captured continuously using `sounddevice`.
2. Each chunk is transformed into the frequency domain using FFT.
3. The spectral amplitudes are mapped to vibrational modes.
4. These modes are sent to a GLSL fragment shader.
5. The shader performs modal superposition.
6. Regions near zero displacement are visualized as nodal lines (sand effect).

---

## Requirements

- Python 3.10+
- GPU supporting OpenGL 3.3+
- NVIDIA / AMD / Integrated GPU

---

## Installation

```bash
pip install sounddevice numpy scipy moderngl moderngl-window
```