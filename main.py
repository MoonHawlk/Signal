"""
Signal - Real-Time Circular Chladni Plate Simulation

This module captures live microphone audio, computes its frequency spectrum
using FFT, and feeds the spectral amplitudes into a real-time OpenGL shader.

The shader simulates a circular vibrating plate using a modal superposition
approach inspired by Chladni patterns.

Pipeline:

Microphone → Audio Buffer → FFT → Mode Amplitudes →
OpenGL Shader → Circular Mask → Nodal Lines Visualization

Author: Filipe Moreno
Project: Signal
"""

import numpy as np
import sounddevice as sd
from scipy.fft import rfft
import moderngl
import moderngl_window as mglw

# ==========================================================
# AUDIO CONFIGURATION
# ==========================================================

SAMPLE_RATE = 44100       # Audio sampling rate (Hz)
CHUNK = 1024              # Number of samples per audio frame
NUM_MODES = 32            # Number of vibrational modes sent to shader

# Global buffer storing latest audio chunk
audio_buffer = np.zeros(CHUNK)


def audio_callback(indata, frames, time, status):
    """
    Audio callback function executed continuously by sounddevice.

    Parameters
    ----------
    indata : ndarray
        Incoming audio samples from microphone.
    frames : int
        Number of frames captured.
    time : CData
        Timing information.
    status : CallbackFlags
        Status of the audio stream.

    Behavior
    --------
    Stores the most recent chunk of audio in a global buffer.
    """
    global audio_buffer
    audio_buffer = indata[:, 0]


# Start real-time microphone stream
stream = sd.InputStream(
    samplerate=SAMPLE_RATE,
    blocksize=CHUNK,
    channels=1,
    callback=audio_callback
)
stream.start()


# ==========================================================
# OPENGL WINDOW CONFIGURATION
# ==========================================================

class ChladniWindow(mglw.WindowConfig):
    """
    Real-time OpenGL window rendering a circular Chladni plate.

    This class:

    - Receives FFT spectrum from live audio
    - Maps spectral bins to modal amplitudes
    - Sends amplitudes to fragment shader
    - Renders circular vibrating plate
    """

    gl_version = (3, 3)
    title = "Chladni Circular - Real Time"
    window_size = (800, 800)

    def __init__(self, **kwargs):
        """
        Initialize OpenGL program, shaders and geometry.
        """
        super().__init__(**kwargs)

        # Shader Program
        self.prog = self.ctx.program(
            vertex_shader='''
                #version 330

                in vec2 in_vert;
                out vec2 uv;

                void main() {
                    uv = in_vert;
                    gl_Position = vec4(in_vert, 0.0, 1.0);
                }
            ''',

            fragment_shader='''
                #version 330

                uniform float modes[32];
                in vec2 uv;
                out vec4 fragColor;

                void main() {

                    vec2 p = uv;

                    // Radial distance from center
                    float r = length(p);

                    // Discard outside circular plate
                    if (r > 1.0)
                        discard;

                    float z = 0.0;

                    // Modal superposition
                    for (int i = 1; i < 32; i++) {
                        float m = mod(i, 8) + 1;
                        float n = mod(i / 4, 8) + 1;

                        z += modes[i] *
                             sin(m * 3.1415 * p.x) *
                             sin(n * 3.1415 * p.y);
                    }

                    // Nodal line detection
                    float nodal = abs(z);

                    // "Sand accumulation" effect
                    float sand = smoothstep(0.02, 0.0, nodal);

                    fragColor = vec4(vec3(sand), 1.0);
                }
            '''
        )

        # Full-screen quad geometry
        vertices = np.array([
            -1, -1,
             1, -1,
            -1,  1,
             1,  1
        ], dtype='f4')

        self.vbo = self.ctx.buffer(vertices)
        self.vao = self.ctx.simple_vertex_array(
            self.prog,
            self.vbo,
            'in_vert'
        )

    def on_render(self, time, frame_time):
        """
        Called every frame by moderngl_window.

        Steps:
        1. Compute FFT of latest audio chunk.
        2. Normalize spectrum.
        3. Send spectral amplitudes to shader.
        4. Render vibrating plate.
        """

        # Compute magnitude spectrum
        fft_vals = np.abs(rfft(audio_buffer))

        # Normalize to prevent division by zero
        fft_vals /= np.max(fft_vals) + 1e-6

        # Prepare mode array
        modes = np.zeros(NUM_MODES, dtype='f4')
        bins = min(NUM_MODES, len(fft_vals))
        modes[:bins] = fft_vals[:bins]

        # Send to GPU
        self.prog['modes'].write(modes.tobytes())

        # Render
        self.ctx.clear(0, 0, 0)
        self.vao.render(moderngl.TRIANGLE_STRIP)


# ==========================================================
# APPLICATION ENTRY POINT
# ==========================================================

if __name__ == "__main__":
    """
    Launch OpenGL window and start real-time simulation.
    """
    mglw.run_window_config(ChladniWindow)