#!/usr/bin/env python3
"""
Image -> Sound: direct sonification with full sound design and a
multi-image gallery (side-by-side thumbnails, drag & drop, clipboard paste).

  * Modes: oscillator bank / waveform scan / raster scan / legacy spectrogram
  * Colour channels: mono, RGB timbre banks, stereo R->L / B->R
  * Per-oscillator pluck envelope (attack/decay/strum = image arpeggios)
  * Effects: drive, resonant lowpass, beat-synced delay, Schroeder reverb
  * Seed lock for reproducible renders
  * Region select per image (drag a rect on the preview)
  * Read time + repeats: sweep speed decoupled from track length (BPM)
  * Multi-image upload: multi-select, folder drop, Ctrl+V paste
  * Transport, WAV + MP4 export (Shorts/YouTube)

Dependencies:
    pip install PySide6 numpy pillow imageio-ffmpeg
"""
import io
import os
import sys
import wave
import tempfile
import subprocess

import numpy as np
from PIL import Image, ImageOps, ImageDraw, ImageFont

try:
    import imageio_ffmpeg
except ImportError:
    imageio_ffmpeg = None

from PySide6.QtCore import Qt, QThread, Signal, QBuffer, QByteArray, QSize
from PySide6.QtGui import (QImage, QPixmap, QShortcut, QKeySequence,
                           QPainter, QPen, QColor, QRect, QIcon, QKeySequence)
from PySide6.QtMultimedia import QMediaPlayer, QAudioOutput
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton, QVBoxLayout,
    QHBoxLayout, QGridLayout, QFormLayout, QFileDialog, QDoubleSpinBox,
    QSpinBox, QComboBox, QCheckBox, QGroupBox, QMessageBox, QSlider,
    QSplitter, QDialog, QLineEdit, QProgressBar, QSizePolicy, QListWidget,
    QListWidgetItem,
)

SAMPLE_RATE = 44100
RESAMPLE = getattr(Image, "Resampling", Image).LANCZOS
IMAGE_FILTER = "Images (*.png *.jpg *.jpeg *.bmp *.gif *.webp *.tif *.tiff)"
IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp", ".tif", ".tiff")
MAX_STORED = 3000          # stored images are downscaled to this max side
THUMB = QSize(140, 105)


def find_ffmpeg():
    if imageio_ffmpeg is not None:
        try:
            return imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:
            pass
    from shutil import which
    return which("ffmpeg")


# ---------------------------------------------------------------- colormap --
_INFERNO = np.array([
    [0, 0, 4], [27, 12, 65], [74, 12, 107], [120, 28, 109], [165, 44, 96],
    [207, 68, 70], [237, 105, 37], [251, 155, 6], [247, 209, 61],
    [252, 255, 164]], dtype=np.float64)


def colormap(values01):
    v = np.clip(values01, 0.0, 1.0) * (len(_INFERNO) - 1)
    i0 = np.floor(v).astype(int)
    i1 = np.minimum(i0 + 1, len(_INFERNO) - 1)
    f = (v - i0)[..., None]
    return (_INFERNO[i0] * (1.0 - f) + _INFERNO[i1] * f).astype(np.uint8)


def map_to_pil(map2d, w, h):
    a = np.asarray(map2d, dtype=np.float64)
    if a.ndim == 2:
        a = a / max(float(a.max()), 1e-9)
        rgb = colormap(np.flipud(np.sqrt(a)))
    else:
        a = np.clip(a / max(float(a.max()), 1e-9), 0.0, 1.0)
        rgb = (np.flipud(a, axis=0) * 255.0).astype(np.uint8)
    return Image.fromarray(rgb, "RGB").resize((int(w), int(h)), RESAMPLE)


def spectrogram_pixmap(mag):
    m = np.asarray(mag, dtype=np.float64)
    m = m / max(m.max(), 1e-9)
    m = np.flipud(np.sqrt(m))
    rgb = colormap(m)
    h, w, _ = rgb.shape
    img = QImage(rgb.tobytes(), w, h, 3 * w, QImage.Format_RGB888)
    return QPixmap.fromImage(img.copy())


def canvas_pixmap(map2d):
    a = np.asarray(map2d, dtype=np.float64)
    if a.ndim == 3:
        a = np.clip(a / max(float(a.max()), 1e-9), 0.0, 1.0)
        rgb = (np.flipud(a, axis=0) * 255.0).astype(np.uint8)
        h, w, _ = rgb.shape
        img = QImage(rgb.tobytes(), w, h, 3 * w, QImage.Format_RGB888)
        return QPixmap.fromImage(img.copy())
    g = (np.flipud(np.clip(a, 0.0, 1.0)) * 255.0).astype(np.uint8)
    h, w = g.shape
    img = QImage(g.tobytes(), w, h, w, QImage.Format_Grayscale8)
    return QPixmap.fromImage(img.copy())


def _fmt_time(ms):
    t = max(0.0, ms / 1000.0)
    m, s = divmod(t, 60.0)
    return f"{int(m)}:{s:04.1f}"


def audio_to_wav_bytes(stereo, sample_rate):
    st = np.clip(np.asarray(stereo), -1.0, 1.0)
    pcm = (st * 32767.0).astype("<i2").reshape(-1)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(st.shape[1])
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())
    return buf.getvalue()


def qimage_to_pil(qimg):
    """Clipboard QImage -> PIL RGB image (handles row padding)."""
    if qimg is None or qimg.isNull():
        return None
    qimg = qimg.convertTo(QImage.Format_RGBA8888)
    w, h = qimg.width(), qimg.height()
    bpl = qimg.bytesPerLine()
    arr = np.frombuffer(qimg.constBits(), dtype=np.uint8)
    arr = arr.reshape(h, bpl)[:, :w * 4].reshape(h, w, 4)
    return Image.fromarray(arr[:, :, :3], "RGB")


def load_pil(path):
    img = Image.open(path)
    if getattr(img, "is_animated", False):
        img.seek(0)
    img = ImageOps.exif_transpose(img) or img
    img.load()
    if max(img.size) > MAX_STORED:
        img = img.copy()
        img.thumbnail((MAX_STORED, MAX_STORED), RESAMPLE)
    return img


# ------------------------------------------------------------- synth engine --
class ImageSynth:
    """Turns a picture DIRECTLY into (stereo) audio."""

    RGB_BANK_TIMBRES = {"R": "saw", "G": "triangle", "B": "sine"}

    def __init__(self, sample_rate=SAMPLE_RATE, n_fft=2048):
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop = n_fft // 4

    def generate(self, image, mode="osc", timbre="sine", channels="mono",
                 bpm=120, beats=16, read_s=None, repeats=1,
                 pluck=False, pluck_div=1.0, attack=0.008, decay=0.35,
                 strum=0.0,
                 pulse=False, pulse_div=1.0, pulse_depth=0.7, click=False,
                 fmin=50, fmax=8000, log_scale=True, gamma=1.4, invert=False,
                 gl_iters=16, osc_rows=256,
                 drive=0.0, cutoff=20000.0, reso=0.7,
                 delay_div=0.0, delay_fb=0.35, delay_mix=0.0,
                 rev_size=0.5, rev_mix=0.0, seed=None):
        sr = self.sample_rate
        used_seed = (int(seed) if seed is not None
                     else int(np.random.default_rng().integers(0, 2 ** 31)))
        rng = np.random.default_rng(used_seed)

        bpm = float(bpm)
        track_len = max(0.1, beats * 60.0 / bpm)
        read_s = track_len if not read_s else float(np.clip(read_s, 0.05, 120.0))
        repeats = int(np.clip(int(repeats), 1, 16))
        delay_t = float(delay_div) * 60.0 / bpm if delay_div else 0.0

        tail = 0.0
        if rev_mix > 0.0:
            tail = max(tail, 0.4 + 2.6 * rev_size)
        if delay_mix > 0.0 and delay_t > 0.0:
            tail = max(tail, min(3.0, delay_t * 3.0 / max(delay_fb, 0.1)))
        n = max(int(round((track_len + tail) * sr)), sr // 10)
        t = np.arange(n, dtype=np.float64) / sr
        osc_rows = int(np.clip(int(osc_rows), 16, 1024))

        if mode == "spec":
            cols = max(8, int(round(track_len * sr / self.hop)))
            canvas = np.flipud(self._canvas(self._source(image).convert("L"),
                                            cols, 512, gamma, invert))
            mag = self._remap_to_bins(canvas, self.n_fft // 2 + 1,
                                      fmin, fmax, log_scale)
            audio = self._magnitudes_to_audio(mag, gl_iters, rng)
            st = np.stack([audio, audio], axis=1)
            audio_map = mag
        else:
            rows = osc_rows if mode == "osc" else 256
            cols = int(np.clip(read_s * 90.0, 32, 1400))
            src = self._source(image)
            if channels == "mono":
                bands = {"L": self._canvas(src.convert("L"), cols, rows,
                                           gamma, invert)}
                timbres, pans = {"L": timbre}, {"L": 0.0}
            else:
                r_, g_, b_ = src.convert("RGB").split()
                bands = {"R": self._canvas(r_, cols, rows, gamma, invert),
                         "G": self._canvas(g_, cols, rows, gamma, invert),
                         "B": self._canvas(b_, cols, rows, gamma, invert)}
                if channels == "rgb":
                    timbres = {k: self.RGB_BANK_TIMBRES[k] for k in "RGB"}
                    pans = {"R": 0.0, "G": 0.0, "B": 0.0}
                else:
                    timbres = {k: timbre for k in "RGB"}
                    pans = {"R": -0.8, "G": 0.0, "B": 0.8}

            st = np.zeros((n, 2))
            pluck_period = pluck_div * 60.0 / bpm if pluck else 0.0
            for key, cv in bands.items():
                if mode == "osc":
                    self._render_osc(st, cv, timbres[key], fmin, fmax,
                                     log_scale, pans[key], t, sr, read_s,
                                     repeats, pluck, pluck_period, attack,
                                     decay, strum, rng)
                elif mode == "wave":
                    self._render_wave(st, cv, pans[key], t, sr, read_s,
                                      repeats, pluck, pluck_period,
                                      attack, decay)
                else:
                    self._render_raster(st, cv, pans[key], t, read_s,
                                        repeats, pluck, pluck_period,
                                        attack, decay)
            audio_map = self._composite_map(bands)

        if click:
            self._add_click(st, sr, bpm, beats)
        if pulse:
            st *= self._pulse_env(t, bpm, pulse_div, pulse_depth)[:, None]
        if pluck and mode == "spec":
            st *= self._pluck_env(t, pluck_div * 60.0 / bpm,
                                  attack, decay)[:, None]

        st = self._fx_chain(st, sr, drive, cutoff, reso, delay_t,
                            delay_fb, delay_mix, rev_size, rev_mix)
        return self._polish(st), audio_map, used_seed

    # ---------------------------------------------------------- canvases --
    @staticmethod
    def _source(image):
        if max(image.size) > 4000:
            s = image.copy()
            s.thumbnail((4000, 4000), RESAMPLE)
            return s
        return image

    @staticmethod
    def _canvas(band_img, cols, rows, gamma, invert):
        g = band_img.resize((int(cols), int(rows)), RESAMPLE)
        a = np.asarray(g, dtype=np.float64) / 255.0
        if invert:
            a = 1.0 - a
        return np.clip(a, 0.0, 1.0) ** max(float(gamma), 0.01)

    @staticmethod
    def _composite_map(bands):
        if set(bands) == {"R", "G", "B"}:
            return np.stack([bands["R"], bands["G"], bands["B"]], axis=-1)
        return next(iter(bands.values()))

    @staticmethod
    def _col_frac(t, read_s, repeats):
        u = t / max(read_s, 1e-6)
        u = np.minimum(u, max(repeats, 1) - 1e-9)
        return u - np.floor(u)

    # -------------------------------------------------------- envelopes --
    @staticmethod
    def _pulse_env(t, bpm, div, depth):
        period = (60.0 / bpm) * div
        phase = (t % period) / period
        return 1.0 - depth + depth * np.exp(-5.0 * phase)

    @staticmethod
    def _pluck_env(t, period, attack, decay):
        ph = t % period
        return np.where(ph < attack, ph / max(attack, 1e-4),
                        np.exp(-(ph - attack) / max(decay, 1e-3)))

    # -------------------------------------------------------- waveforms --
    @staticmethod
    def _blep_saw(p, dt):
        y = 2.0 * p - 1.0
        pb = np.zeros_like(p)
        dt = max(dt, 1e-9)
        m1 = p < dt
        if m1.any():
            q = p[m1] / dt
            pb[m1] = q + q - q * q - 1.0
        m2 = p > 1.0 - dt
        if m2.any():
            q = (p[m2] - 1.0) / dt
            pb[m2] = q * q + q + q + 1.0
        return y - pb

    @classmethod
    def _wave(cls, kind, p, dt):
        if kind == "sine":
            return np.sin(2.0 * np.pi * p)
        if kind == "triangle":
            return 4.0 * np.abs(p - 0.5) - 1.0
        saw = cls._blep_saw(p, dt)
        if kind == "saw":
            return saw
        return saw - cls._blep_saw((p + 0.5) % 1.0, dt)

    # -------------------------------------------------------- renderers --
    def _render_osc(self, st, canvas, timbre, fmin, fmax, log_scale, pan,
                    t, sr, read_s, repeats, pluck, period, attack, decay,
                    strum, rng):
        rows, cols = canvas.shape
        fmin = max(float(fmin), 1.0)
        fmax = max(float(fmax), fmin * 2.0)
        pos = np.linspace(0.0, 1.0, rows)
        freqs = (fmin * (fmax / fmin) ** pos if log_scale
                 else fmin + (fmax - fmin) * pos)
        phases = rng.uniform(0.0, 2.0 * np.pi, rows)
        colpos = self._col_frac(t, read_s, repeats)
        cx = np.linspace(0.0, 1.0, cols)
        gl, gr = 0.5 - 0.5 * pan, 0.5 + 0.5 * pan
        offsets = strum * period * pos if (pluck and period > 0) else None
        act = canvas.max(axis=1) >= 0.004
        gain = 1.0 / np.sqrt(max(1, int(act.sum())))
        for r in np.nonzero(act)[0]:
            amp = np.interp(colpos, cx, canvas[r])
            if pluck:
                ph = (t - offsets[r]) % period
                amp *= np.where(ph < attack, ph / max(attack, 1e-4),
                                np.exp(-(ph - attack) / max(decay, 1e-3)))
            f = freqs[r]
            p = (phases[r] + 2.0 * np.pi * f * t) / (2.0 * np.pi)
            p -= np.floor(p)
            w = self._wave(timbre, p, f / sr) * (amp * gain)
            st[:, 0] += w * gl
            st[:, 1] += w * gr

    def _render_wave(self, st, canvas, pan, t, sr, read_s, repeats,
                     pluck, period, attack, decay):
        rows, cols = canvas.shape
        w = canvas * 2.0 - 1.0
        w -= w.mean(axis=0, keepdims=True)
        pk = np.max(np.abs(w), axis=0, keepdims=True)
        pk[pk < 1e-6] = 1.0
        w /= pk
        colpos = self._col_frac(t, read_s, repeats)
        ci = colpos * (cols - 1)
        col_s = read_s / cols
        ri = ((t / max(col_s, 1e-6)) % 1.0) * rows
        r0 = np.floor(ri).astype(int) % rows
        r1 = (r0 + 1) % rows
        fr = ri - np.floor(ri)
        c0 = np.clip(np.floor(ci).astype(int), 0, cols - 1)
        c1 = np.minimum(c0 + 1, cols - 1)
        fc = ci - c0
        v = (w[r0, c0] * (1 - fr) * (1 - fc) + w[r1, c0] * fr * (1 - fc)
             + w[r0, c1] * (1 - fr) * fc + w[r1, c1] * fr * fc)
        if pluck:
            v *= self._pluck_env(t, period, attack, decay)
        st[:, 0] += v * (0.5 - 0.5 * pan)
        st[:, 1] += v * (0.5 + 0.5 * pan)

    def _render_raster(self, st, canvas, pan, t, read_s, repeats,
                       pluck, period, attack, decay):
        v = np.flipud(canvas).reshape(-1) * 2.0 - 1.0
        v = v - v.mean()
        pk = np.max(np.abs(v))
        if pk > 1e-9:
            v /= pk
        idx = self._col_frac(t, read_s, repeats) * (v.size - 1)
        val = np.interp(idx, np.arange(v.size), v)
        if pluck:
            val *= self._pluck_env(t, period, attack, decay)
        st[:, 0] += val * (0.5 - 0.5 * pan)
        st[:, 1] += val * (0.5 + 0.5 * pan)

    # ----------------------------------------------------------- click --
    def _add_click(self, st, sr, bpm, beats):
        tt = np.arange(int(0.05 * sr)) / sr
        for b in range(beats):
            s = int(round(b * 60.0 / bpm * sr))
            if s >= st.shape[0]:
                break
            acc = (b % 4) == 0
            f = 2000.0 if acc else 1400.0
            a = 0.5 if acc else 0.3
            tick = np.sin(2.0 * np.pi * f * tt) * np.exp(-tt * 90.0) * a
            e = min(s + tt.size, st.shape[0])
            st[s:e, 0] += tick[:e - s]
            st[s:e, 1] += tick[:e - s]

    # --------------------------------------------------------- effects --
    def _fx_chain(self, st, sr, drive, cutoff, reso, delay_t,
                  delay_fb, delay_mix, rev_size, rev_mix):
        if drive > 0.0:
            g = 1.0 + drive * 9.0
            st = np.tanh(g * st) / np.tanh(g)
        if cutoff < 20000.0:
            st = self._lowpass(st, sr, cutoff, reso)
        if delay_mix > 0.0 and delay_t > 0.0:
            st = self._delay(st, sr, delay_t, delay_fb, delay_mix)
        if rev_mix > 0.0:
            st = self._reverb(st, sr, rev_size, rev_mix)
        return st

    def _lowpass(self, st, sr, fc, q):
        fc = min(float(fc), 0.45 * sr)
        q = max(float(q), 0.1)
        n = st.shape[0]
        f = np.fft.rfftfreq(n, 1.0 / sr)
        w = 2.0 * np.pi * f / sr
        w0 = 2.0 * np.pi * fc / sr
        alpha = np.sin(w0) / (2.0 * q)
        cw = np.cos(w0)
        b0 = (1 - cw) / 2
        b1 = 1 - cw
        a0 = 1 + alpha
        a1 = -2 * cw
        a2 = 1 - alpha
        z = np.exp(-1j * w)
        H = (b0 + b1 * z + b0 * z * z) / (a0 + a1 * z + a2 * z * z)
        return np.fft.irfft(np.fft.rfft(st, axis=0) * np.abs(H)[:, None],
                            n, axis=0)

    @staticmethod
    def _delay(st, sr, time_s, fb, mix):
        n = st.shape[0]
        d = int(round(time_s * sr))
        if d < 8 or d >= n:
            return st
        wet = np.zeros_like(st)
        gain, k = 1.0, d
        while k < n and gain > 0.005:
            wet[k:] += st[:n - k] * gain
            gain *= max(float(fb), 0.0)
            k += d
        return st * (1.0 - mix) + wet * mix

    def _reverb(self, st, sr, size, mix):
        ir = self._schroeder_ir(sr, size, st.shape[0])
        wet = np.stack([self._fft_conv(st[:, c], ir[:, c]) for c in (0, 1)],
                       axis=1)
        wpk = np.max(np.abs(wet))
        if wpk > 1e-9:
            wet *= 0.9 * np.max(np.abs(st)) / wpk
        return st * (1.0 - mix) + wet * mix

    def _schroeder_ir(self, sr, size, min_len):
        n = max(int((0.4 + 2.6 * size) * sr), min_len // 2, sr // 4)
        ir = np.zeros((n, 2))
        g = 0.62 + 0.25 * size
        for ch, det in ((0, 1.0), (1, 1.013)):
            for d0 in (1310, 1636, 1813, 1927):
                d = max(64, int(d0 * sr / 44100.0 * det * (0.85 + 0.3 * size)))
                k = np.arange(0, n, d)
                ir[k, ch] += g ** (k / d)
            rng = np.random.default_rng(7 + ch)
            for _ in range(6):
                pos = int(rng.uniform(0.003, 0.08) * sr)
                if pos < n:
                    ir[pos, ch] += rng.uniform(0.2, 0.5)
            env = np.exp(-np.arange(n) / (0.5 + 1.6 * size) / sr)
            ir[:, ch] += rng.normal(0.0, 1.0, n) * env * 0.02
        pk = np.max(np.abs(ir))
        if pk > 0:
            ir /= pk
        return ir

    @staticmethod
    def _fft_conv(x, ir):
        n = x.size + ir.size - 1
        return np.fft.irfft(np.fft.rfft(x, n) * np.fft.rfft(ir, n),
                            n)[:x.size]

    def _polish(self, st, peak=0.9):
        st = np.asarray(st, dtype=np.float64)
        m = np.max(np.abs(st))
        if m > 0:
            st *= peak / m
        nf = min(st.shape[0] // 2, int(0.015 * self.sample_rate))
        if nf > 0:
            ramp = np.linspace(0.0, 1.0, nf)[:, None]
            st[:nf] *= ramp
            st[-nf:] *= ramp[::-1]
        return st

    # ------------------------------------------------- legacy spectrogram --
    def _remap_to_bins(self, canvas, n_bins, fmin, fmax, log_scale):
        fmin = max(float(fmin), 1.0)
        fmax = max(float(fmax), fmin * 2.0)
        freqs = np.fft.rfftfreq(self.n_fft, 1.0 / self.sample_rate)
        if log_scale:
            pos = np.log(np.clip(freqs, fmin, fmax) / fmin) / np.log(fmax / fmin)
        else:
            pos = (freqs - fmin) / (fmax - fmin)
        pos = np.clip(pos, 0.0, 1.0)
        r = pos * (canvas.shape[0] - 1)
        r0 = np.floor(r).astype(int)
        r1 = np.minimum(r0 + 1, canvas.shape[0] - 1)
        frac = (r - r0)[:, None]
        mag = canvas[r0, :] * (1.0 - frac) + canvas[r1, :] * frac
        mag[(freqs < fmin) | (freqs > fmax), :] = 0.0
        return mag

    def _magnitudes_to_audio(self, mag, gl_iters, rng):
        n = self.n_fft
        window = 0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(n) / n)
        hop = self.hop
        length = (mag.shape[1] - 1) * hop + n

        def stft(x, n_frames):
            need = (n_frames - 1) * hop + n
            x = np.pad(x, (0, max(0, need - x.size)))[:need]
            fr = np.lib.stride_tricks.sliding_window_view(x, n)[::hop]
            return np.fft.rfft(fr * window, axis=1).T

        def istft(spec, ln):
            fr = np.fft.irfft(spec, n=n, axis=0).T
            total = (spec.shape[1] - 1) * hop + n
            out = np.zeros(total)
            wsum = np.zeros(total)
            for k in range(spec.shape[1]):
                s = k * hop
                out[s:s + n] += fr[k] * window
                wsum[s:s + n] += window * window
            nz = wsum > 1e-10
            out[nz] /= wsum[nz]
            return out[:ln]

        spec = mag * np.exp(1j * rng.uniform(0, 2 * np.pi, mag.shape))
        for _ in range(int(gl_iters)):
            x = istft(spec, length)
            spec = mag * np.exp(1j * np.angle(stft(x, mag.shape[1])))
        return istft(spec, length)


# ---------------------------------------------------------- region select --
class RegionImageLabel(QLabel):
    region_changed = Signal(object)

    def __init__(self, text="", parent=None):
        super().__init__(text, parent)
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(320, 240)
        self.setStyleSheet("background:#101014; color:#777;")
        self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.setMouseTracking(True)
        self._full = None
        self._scaled = None
        self._lw = self._lh = 0
        self._region = None
        self._drag = None
        self._geo = None

    def set_image(self, pixmap, region=None):
        self._full = pixmap
        self._scaled = None
        self._region = QRect(region) if region is not None else None
        self._drag = None
        self._refresh()

    def image_region(self):
        return QRect(self._region) if self._region is not None else None

    def clear_region(self):
        if self._region is not None or self._drag is not None:
            self._drag = None
            self._region = None
            self._refresh()
            self.region_changed.emit(None)

    def _refresh(self):
        if self._full is None:
            self._geo = None
            self.setPixmap(QPixmap())
            self.setText("no image — add one (📂 / Ctrl+V / drag & drop)")
            return
        lw, lh = max(self.width(), 60), max(self.height(), 60)
        if self._scaled is None or self._lw != lw or self._lh != lh:
            self._scaled = self._full.scaled(lw, lh, Qt.KeepAspectRatio,
                                             Qt.SmoothTransformation)
            self._lw, self._lh = lw, lh
        dx = (lw - self._scaled.width()) // 2
        dy = (lh - self._scaled.height()) // 2
        sx = self._full.width() / max(self._scaled.width(), 1)
        sy = self._full.height() / max(self._scaled.height(), 1)
        self._geo = (dx, dy, sx, sy)

        base = QPixmap(lw, lh)
        base.fill(QColor("#101014"))
        p = QPainter(base)
        p.drawPixmap(dx, dy, self._scaled)
        rect = self._drag
        if rect is None and self._region is not None:
            dxg, dyg, sxg, syg = self._geo
            rect = QRect(dxg + int(self._region.x() / sxg),
                         dyg + int(self._region.y() / syg),
                         max(1, int(self._region.width() / sxg)),
                         max(1, int(self._region.height() / syg)))
        if rect is not None and rect.width() > 1 and rect.height() > 1:
            r = rect.normalized()
            p.fillRect(0, 0, lw, r.top(), QColor(0, 0, 0, 150))
            p.fillRect(0, r.bottom() + 1, lw, lh - r.bottom() - 1,
                       QColor(0, 0, 0, 150))
            p.fillRect(0, r.top(), r.left(), r.height(), QColor(0, 0, 0, 150))
            p.fillRect(r.right() + 1, r.top(), lw - r.right() - 1, r.height(),
                       QColor(0, 0, 0, 150))
            p.setPen(QPen(QColor(255, 214, 92), 2, Qt.DashLine))
            p.drawRect(r)
            p.setPen(QPen(QColor(255, 214, 92)))
            p.drawText(r.left() + 6, r.top() + 18,
                       f"{int(r.width() * sx)}×{int(r.height() * sy)} px")
        p.end()
        self.setPixmap(base)

    def _widget_to_image(self, rect):
        if self._geo is None or self._full is None:
            return None
        dx, dy, sx, sy = self._geo
        x0 = (rect.left() - dx) * sx
        y0 = (rect.top() - dy) * sy
        x1 = (rect.right() + 1 - dx) * sx
        y1 = (rect.bottom() + 1 - dy) * sy
        r = QRect(int(round(min(x0, x1))), int(round(min(y0, y1))),
                  int(round(abs(x1 - x0))), int(round(abs(y1 - y0))))
        r = r.intersected(QRect(0, 0, self._full.width(), self._full.height()))
        return r if r.width() >= 8 and r.height() >= 8 else None

    def mousePressEvent(self, ev):
        if self._full is None:
            return
        if ev.button() == Qt.LeftButton:
            self._drag = QRect(ev.position().toPoint(), QSize(0, 0))
        elif ev.button() in (Qt.RightButton, Qt.MiddleButton):
            self.clear_region()

    def mouseMoveEvent(self, ev):
        if self._drag is not None:
            self._drag = QRect(self._drag.topLeft(),
                               ev.position().toPoint()).normalized()
            self._refresh()

    def mouseReleaseEvent(self, ev):
        if self._drag is None:
            return
        drag, self._drag = self._drag, None
        self._region = self._widget_to_image(drag)
        self._refresh()
        self.region_changed.emit(QRect(self._region)
                                 if self._region is not None else None)

    def mouseDoubleClickEvent(self, ev):
        self.clear_region()

    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        self._refresh()


# ------------------------------------------------------- NEW: image gallery --
class ImageGallery(QListWidget):
    """Side-by-side thumbnail strip of every loaded image."""
    remove_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setViewMode(QListWidget.IconMode)
        self.setFlow(QListWidget.LeftToRight)
        self.setWrapping(False)
        self.setSelectionMode(QListWidget.SingleSelection)
        self.setMovement(QListWidget.Static)
        self.setIconSize(THUMB)
        self.setSpacing(8)
        self.setUniformItemSizes(True)
        self.setFixedHeight(THUMB.height() + 58)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setStyleSheet("""
            QListWidget { background:#14141a; border:1px solid #bbb;
                          border-radius:6px; outline:0; }
            QListWidget::item { border:2px solid transparent; border-radius:6px;
                                color:#ccc; font-size:10px; padding:2px; }
            QListWidget::item:selected { border:2px solid #ffd65c;
                                         background:#2a2a33; color:#fff; }
            QListWidget::item:hover { border:2px solid #888; }
        """)

    def keyPressEvent(self, ev):
        if ev.key() in (Qt.Key_Delete, Qt.Key_Backspace):
            self.remove_requested.emit()
        else:
            super().keyPressEvent(ev)


# ------------------------------------------------------------ video export --
_FONT_CANDIDATES = [
    "C:/Windows/Fonts/segoeui.ttf", "C:/Windows/Fonts/arial.ttf",
    "/System/Library/Fonts/Helvetica.ttc", "/System/Library/Fonts/SFNS.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/dejavu/DejaVuSans.ttf",
]


def load_font(size):
    for path in _FONT_CANDIDATES:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, int(size))
            except Exception:
                pass
    return ImageFont.load_default()


def _fit_pil(source, max_w, max_h):
    img = source.convert("RGB").copy()
    img.thumbnail((max(1, int(max_w)), max(1, int(max_h))), RESAMPLE)
    return img


class VideoComposer:
    BG = (13, 13, 18)
    TEXT = (240, 240, 248)
    PLAYHEAD = (255, 214, 92)

    def __init__(self, source, map2d, size, mode="image_sweep", caption="",
                 scan_windows=None, duration=1.0):
        self.w, self.h = int(size[0]), int(size[1])
        self.mode = mode
        self.windows = [tuple(x) for x in (scan_windows or [])]
        self.duration = max(float(duration), 1e-6)

        bg = Image.new("RGB", (self.w, self.h), self.BG)
        draw = ImageDraw.Draw(bg)
        top = int(self.h * 0.035)
        if caption:
            font = load_font(max(30, self.w // 22))
            bbox = draw.textbbox((0, 0), caption, font=font)
            tw = bbox[2] - bbox[0]
            draw.text(((self.w - tw) // 2 - bbox[0], top), caption,
                      font=font, fill=self.TEXT,
                      stroke_width=max(2, self.w // 640), stroke_fill=(0, 0, 0))
            top += (bbox[3] - bbox[1]) + int(self.h * 0.04)

        margin = int(self.w * 0.04)
        x0, y0 = margin, top
        x1, y1 = self.w - margin, self.h - int(self.h * 0.035)
        aw, ah = max(4, x1 - x0), max(4, y1 - y0)

        self.static_img = self.static_pos = None
        self.sweep_img = self.sweep_pos = None

        if mode == "image":
            im = _fit_pil(source, aw, ah)
            self.static_img = im
            self.static_pos = (x0 + (aw - im.width) // 2,
                               y0 + (ah - im.height) // 2)
        elif mode == "image_sweep":
            im = _fit_pil(source, aw, ah)
            self.sweep_img = im
            self.sweep_pos = (x0 + (aw - im.width) // 2,
                              y0 + (ah - im.height) // 2)
        elif mode == "combo":
            img_h = int(ah * 0.55)
            gap = int(ah * 0.05)
            im = _fit_pil(source, aw, img_h)
            self.static_img = im
            self.static_pos = (x0 + (aw - im.width) // 2,
                               y0 + (img_h - im.height) // 2)
            sy = y0 + img_h + gap
            self.sweep_img = map_to_pil(map2d, aw, y1 - sy)
            self.sweep_pos = (x0, sy)
        else:
            self.sweep_img = map_to_pil(map2d, aw, ah)
            self.sweep_pos = (x0, y0)
        self.bg = bg

    def _draw(self, frac, show_playhead):
        frame = self.bg.copy()
        if self.sweep_img is not None:
            sw, sh = self.sweep_img.size
            sx, sy = self.sweep_pos
            reveal = int(round(min(max(frac, 0.0), 1.0) * sw))
            if reveal > 0:
                frame.paste(self.sweep_img.crop((0, 0, reveal, sh)), (sx, sy))
            if show_playhead:
                d = ImageDraw.Draw(frame)
                x = min(sx + reveal, sx + sw - 4)
                d.rectangle([x, sy, x + 3, sy + sh - 1], fill=self.PLAYHEAD)
        if self.static_img is not None:
            frame.paste(self.static_img, self.static_pos)
        return np.asarray(frame, dtype=np.uint8)

    def frame(self, i, total):
        t = (i + 1) / max(total, 1) * self.duration
        frac, show = 0.0, False
        for a, b in self.windows:
            if a - 1e-9 <= t < b:
                frac = (t - a) / max(b - a, 1e-6)
                show = True
                break
        if not self.windows:
            frac = 1.0
        return self._draw(frac, show)


class ExportWorker(QThread):
    progress = Signal(int)
    done_ok = Signal(str)
    failed = Signal(str)

    def __init__(self, ffmpeg_exe, composer, wav_bytes, out_path, fps,
                 duration, parent=None):
        super().__init__(parent)
        self.ffmpeg_exe = ffmpeg_exe
        self.composer = composer
        self.wav_bytes = wav_bytes
        self.out_path = out_path
        self.fps = fps
        self.duration = duration

    def run(self):
        try:
            total = max(1, int(round(self.duration * self.fps)))
            with tempfile.TemporaryDirectory() as td:
                wav_path = os.path.join(td, "audio.wav")
                with open(wav_path, "wb") as fh:
                    fh.write(self.wav_bytes)
                err_path = os.path.join(td, "ffmpeg.log")
                cmd = [self.ffmpeg_exe, "-y",
                       "-f", "rawvideo", "-pix_fmt", "rgb24",
                       "-s", f"{self.composer.w}x{self.composer.h}",
                       "-r", str(self.fps), "-i", "-",
                       "-i", wav_path,
                       "-c:v", "libx264", "-preset", "medium",
                       "-crf", "18", "-pix_fmt", "yuv420p",
                       "-movflags", "+faststart",
                       "-c:a", "aac", "-b:a", "192k",
                       "-shortest", self.out_path]
                with open(err_path, "wb") as errf:
                    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                                            stdout=subprocess.DEVNULL,
                                            stderr=errf)
                    try:
                        for i in range(total):
                            if self.isInterruptionRequested():
                                proc.kill()
                                raise RuntimeError("Export cancelled.")
                            proc.stdin.write(
                                self.composer.frame(i, total).tobytes())
                            if i % 4 == 0 or i == total - 1:
                                self.progress.emit(int(100 * (i + 1) / total))
                        proc.stdin.close()
                        rc = proc.wait()
                    except BrokenPipeError:
                        rc = proc.wait()
                if rc != 0:
                    with open(err_path, "r", errors="ignore") as fh:
                        msg = fh.read()[-1600:]
                    raise RuntimeError(msg or f"ffmpeg exited with code {rc}")
            self.done_ok.emit(self.out_path)
        except Exception as exc:
            self.failed.emit(str(exc))


class ExportDialog(QDialog):
    FORMATS = [
        ("YouTube Shorts — 1080×1920 (9:16)", 1080, 1920),
        ("YouTube 1080p — 1920×1080 (16:9)", 1920, 1080),
        ("YouTube 720p — 1280×720 (16:9)", 1280, 720),
        ("Square — 1080×1080 (1:1)", 1080, 1080),
    ]
    VISUALS = [("Image sweep + playhead", "image_sweep"),
               ("Image + synthesis map", "combo"),
               ("Synthesis map + playhead", "map"),
               ("Image only (static)", "image")]

    def __init__(self, main_window, ffmpeg_exe, parent=None):
        super().__init__(parent or main_window)
        self.main = main_window
        self.ffmpeg_exe = ffmpeg_exe
        self.worker = None

        self.setWindowTitle("Export MP4")
        self.setMinimumWidth(580)
        lay = QVBoxLayout(self)

        self.preview = QLabel("preview")
        self.preview.setAlignment(Qt.AlignCenter)
        self.preview.setMinimumHeight(280)
        self.preview.setStyleSheet("background:#0d0d12; color:#666;")
        lay.addWidget(self.preview)

        form = QGridLayout()
        form.setHorizontalSpacing(10)
        form.addWidget(QLabel("Format"), 0, 0)
        self.fmt = QComboBox()
        for name, w, h in self.FORMATS:
            self.fmt.addItem(name, (w, h))
        form.addWidget(self.fmt, 0, 1, 1, 3)

        form.addWidget(QLabel("Visual"), 1, 0)
        self.vis = QComboBox()
        for label, _m in self.VISUALS:
            self.vis.addItem(label)
        form.addWidget(self.vis, 1, 1, 1, 3)

        form.addWidget(QLabel("Frame rate"), 2, 0)
        self.fps = QComboBox()
        self.fps.addItems(["24", "30", "60"])
        self.fps.setCurrentText("30")
        form.addWidget(self.fps, 2, 1)
        self.info = QLabel()
        form.addWidget(self.info, 2, 2, 1, 2)

        form.addWidget(QLabel("Caption"), 3, 0)
        self.caption = QLineEdit()
        self.caption.setPlaceholderText("optional on-screen title text")
        # pre-fill with the selected image's name
        name = getattr(self.main, "image_name", "") or ""
        if name:
            self.caption.setText(os.path.splitext(name)[0])
        form.addWidget(self.caption, 3, 1, 1, 3)

        form.addWidget(QLabel("Output"), 4, 0)
        self.path = QLineEdit()
        self.browse = QPushButton("Browse…")
        form.addWidget(self.path, 4, 1, 1, 2)
        form.addWidget(self.browse, 4, 3)
        lay.addLayout(form)

        self.prog = QProgressBar()
        self.prog.setRange(0, 100)
        lay.addWidget(self.prog)

        btns = QHBoxLayout()
        btns.addStretch(1)
        self.btn_export = QPushButton("⬇ Export MP4")
        self.btn_export.setStyleSheet("font-weight:600;")
        self.btn_cancel = QPushButton("Close")
        btns.addWidget(self.btn_export)
        btns.addWidget(self.btn_cancel)
        lay.addLayout(btns)

        self.fmt.currentIndexChanged.connect(self._refresh)
        self.vis.currentIndexChanged.connect(self._refresh)
        self.fps.currentIndexChanged.connect(self._refresh)
        self.caption.editingFinished.connect(self._refresh)
        self.browse.clicked.connect(self._browse)
        self.btn_export.clicked.connect(self._export)
        self.btn_cancel.clicked.connect(self._cancel)
        self._refresh()

    def _mode(self):
        return self.VISUALS[self.vis.currentIndex()][1]

    def _composer(self):
        w, h = self.fmt.currentData()
        return VideoComposer(self.main.image, self.main.map2d, (w, h),
                             self._mode(), self.caption.text().strip(),
                             getattr(self.main, "scan_windows", []),
                             self.main.duration)

    def _refresh(self):
        fps = int(self.fps.currentText())
        frames = max(1, int(round(self.main.duration * fps)))
        self.info.setText(f"{self.main.duration:.2f} s → {frames} frames")
        try:
            comp = self._composer()
        except Exception:
            return
        arr = comp._draw(0.5, bool(comp.windows))
        img = QImage(arr.tobytes(), comp.w, comp.h, 3 * comp.w,
                     QImage.Format_RGB888)
        self.preview.setPixmap(
            QPixmap.fromImage(img).scaledToHeight(280, Qt.SmoothTransformation))

    def _browse(self):
        stem = os.path.splitext(getattr(self.main, "image_name", "") or
                                "image_sound")[0]
        default = os.path.join(os.path.expanduser("~"), f"{stem}_shorts.mp4")
        path, _ = QFileDialog.getSaveFileName(self, "Save MP4", default,
                                              "MP4 video (*.mp4)")
        if path:
            if not path.lower().endswith(".mp4"):
                path += ".mp4"
            self.path.setText(path)

    def _set_running(self, running):
        for wdg in (self.fmt, self.vis, self.fps, self.caption,
                    self.path, self.browse):
            wdg.setEnabled(not running)
        self.btn_export.setEnabled(not running)
        self.btn_cancel.setText("Cancel" if running else "Close")

    def _export(self):
        if self.worker is not None and self.worker.isRunning():
            return
        out = self.path.text().strip()
        if not out:
            self._browse()
            out = self.path.text().strip()
        if not out:
            return
        if not out.lower().endswith(".mp4"):
            out += ".mp4"
            self.path.setText(out)
        self._set_running(True)
        self.prog.setValue(0)
        fps = int(self.fps.currentText())
        self.worker = ExportWorker(self.ffmpeg_exe, self._composer(),
                                   self.main.wav_bytes, out, fps,
                                   self.main.duration, parent=self)
        self.worker.progress.connect(self.prog.setValue)
        self.worker.done_ok.connect(self._on_done)
        self.worker.failed.connect(self._on_failed)
        self.worker.start()

    def _on_done(self, path):
        self.prog.setValue(100)
        self._set_running(False)
        QMessageBox.information(self, "Export complete",
                                f"MP4 saved:\n{path}\n\nReady to upload.")
        self.accept()

    def _on_failed(self, msg):
        self._set_running(False)
        QMessageBox.critical(self, "Export failed", msg)

    def _cancel(self):
        if self.worker is not None and self.worker.isRunning():
            self.worker.requestInterruption()
            self.worker.wait(5000)
        self.reject()


# ------------------------------------------------------------------ worker --
class SynthWorker(QThread):
    finished_ok = Signal(object, object, int)
    failed = Signal(str)

    def __init__(self, synth, image, params, parent=None):
        super().__init__(parent)
        self.synth, self.image, self.params = synth, image, params

    def run(self):
        try:
            audio, audio_map, used = self.synth.generate(self.image,
                                                         **self.params)
            self.finished_ok.emit(audio, audio_map, int(used))
        except Exception as exc:
            self.failed.emit(str(exc))


# ------------------------------------------------------------- main window --
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Image → Sound  (Sound-Design Sonification)")
        self.resize(1400, 1000)
        self.setAcceptDrops(True)

        self.synth = ImageSynth()
        self.images = []            # NEW: list of dicts {path,name,pil,region}
        self.image = None           # currently selected PIL image
        self.image_name = ""        # NEW: selected image display name
        self.audio = None
        self.map2d = None
        self.mode = "osc"
        self.duration = 0.0
        self.wav_bytes = None
        self.worker = None
        self.scan_windows = []
        self._busy = False
        self._scrubbing = False
        self._paste_count = 0

        self.player = QMediaPlayer(self)
        self.audio_out = QAudioOutput(self)
        self.audio_out.setVolume(0.9)
        self.player.setAudioOutput(self.audio_out)
        self.player.playbackStateChanged.connect(self._on_playback_state)
        self.player.positionChanged.connect(self._on_position)
        self.player.durationChanged.connect(self._on_duration)
        self.player.mediaStatusChanged.connect(self._on_media_status)
        self.player.errorOccurred.connect(
            lambda e, s: self.statusBar().showMessage(f"Player: {s}", 6000))
        self.buffer = None

        self._build_ui()
        self._make_shortcuts()
        self.statusBar().showMessage(
            "Add images: 📂 button, drag & drop (files or a folder), or "
            "Ctrl+V to paste. Click a thumbnail to select. Space = play.")

    # ---------------------------------------------------------------- UI --
    def _build_ui(self):
        central = QWidget(self)
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(10, 10, 10, 6)
        root.setSpacing(8)

        # top bar
        bar = QHBoxLayout()
        self.btn_add = QPushButton("📂 Add Images…")
        self.btn_synth = QPushButton("🎹 Synthesize")
        self.btn_save = QPushButton("💾 Save WAV…")
        self.btn_mp4 = QPushButton("🎬 Export MP4…")
        self.btn_clear_region = QPushButton("🔲 Clear region")
        self.btn_clear_region.setEnabled(False)
        for b in (self.btn_add, self.btn_synth, self.btn_save, self.btn_mp4,
                  self.btn_clear_region):
            b.setFocusPolicy(Qt.TabFocus)
            bar.addWidget(b)
        bar.addStretch(1)
        root.addLayout(bar)

        self.btn_add.clicked.connect(self.add_images_dialog)
        self.btn_synth.clicked.connect(self.synthesize)
        self.btn_save.clicked.connect(self.save_wav)
        self.btn_mp4.clicked.connect(self.open_export)
        self.btn_clear_region.clicked.connect(self._clear_region)

        # transport
        transport = QGroupBox("Transport")
        tlay = QHBoxLayout(transport)
        tlay.setSpacing(6)

        def tbtn(text, tip):
            b = QPushButton(text)
            b.setToolTip(tip)
            b.setFocusPolicy(Qt.TabFocus)
            b.setMinimumWidth(64)
            return b

        self.btn_rewind = tbtn("⏮", "Back to start  (Home)")
        self.btn_play = tbtn("▶ Play", "Play / Pause  (Space)")
        self.btn_stop = tbtn("⏹ Stop", "Stop and rewind  (S)")
        self.btn_play.setMinimumWidth(90)
        self.btn_play.setStyleSheet("font-weight:600;")
        tlay.addWidget(self.btn_rewind)
        tlay.addWidget(self.btn_play)
        tlay.addWidget(self.btn_stop)
        self.loop = QCheckBox("Loop")
        self.loop.setFocusPolicy(Qt.TabFocus)
        tlay.addWidget(self.loop)
        self.time_label = QLabel("0:00.0 / 0:00.0")
        self.time_label.setStyleSheet("font-family:Consolas,monospace;")
        self.seek = QSlider(Qt.Horizontal)
        self.seek.setRange(0, 0)
        self.seek.setEnabled(False)
        tlay.addWidget(self.time_label)
        tlay.addWidget(self.seek, 1)
        tlay.addWidget(QLabel("Vol"))
        self.vol = QSlider(Qt.Horizontal)
        self.vol.setRange(0, 100)
        self.vol.setValue(90)
        self.vol.setFixedWidth(110)
        self.vol.valueChanged.connect(lambda v: self.audio_out.setVolume(v / 100))
        tlay.addWidget(self.vol)
        root.addWidget(transport)

        self.btn_rewind.clicked.connect(self.rewind)
        self.btn_play.clicked.connect(self.toggle_play_pause)
        self.btn_stop.clicked.connect(self.stop_playback)
        self.loop.toggled.connect(lambda _: None)
        self.seek.sliderPressed.connect(self._seek_pressed)
        self.seek.sliderReleased.connect(self._seek_released)
        self.seek.sliderMoved.connect(self.player.setPosition)

        # --- NEW: side-by-side image gallery strip ---
        gal_box = QGroupBox()
        gal_lay = QHBoxLayout(gal_box)
        gal_lay.setContentsMargins(6, 2, 6, 4)
        self.gallery = ImageGallery()
        self.gallery.setAcceptDrops(False)          # main window handles drops
        self.gallery.itemSelectionChanged.connect(self._on_gallery_select)
        self.gallery.itemDoubleClicked.connect(
            lambda _item: self.synthesize())
        self.gallery.remove_requested.connect(self.remove_selected_image)
        gal_lay.addWidget(self.gallery, 1)

        gal_btns = QVBoxLayout()
        gal_btns.setSpacing(4)
        self.gal_count = QLabel("0 images")
        self.gal_count.setAlignment(Qt.AlignCenter)
        self.gal_count.setStyleSheet("font-weight:600;")
        self.btn_remove_img = QPushButton("🗑 Remove")
        self.btn_remove_img.setToolTip("Remove selected image (Del)")
        self.btn_clear_imgs = QPushButton("🧹 Clear all")
        for w in (self.gal_count, self.btn_remove_img, self.btn_clear_imgs):
            gal_btns.addWidget(w)
        gal_btns.addStretch(1)
        gal_lay.addLayout(gal_btns)
        self.btn_remove_img.clicked.connect(self.remove_selected_image)
        self.btn_clear_imgs.clicked.connect(self.clear_all_images)
        root.addWidget(gal_box)

        # previews (source + synthesis map, side by side)
        split = QSplitter(Qt.Horizontal)
        root.addWidget(split, 1)

        def preview_box(title, label_widget):
            group = QGroupBox(title)
            lay = QVBoxLayout(group)
            lay.addWidget(label_widget)
            split.addWidget(group)
            return group

        self.image_view = RegionImageLabel("no image")
        self.image_view.setAcceptDrops(False)
        self.image_view.region_changed.connect(self._on_region)
        preview_box("Source image  — drag = region, right-click = clear "
                    "(X = time, Y = pitch, bright = loud)", self.image_view)

        self.spec_view = QLabel("press Synthesize")
        self.spec_view.setAlignment(Qt.AlignCenter)
        self.spec_view.setMinimumSize(320, 240)
        self.spec_view.setStyleSheet("background:#101014; color:#777;")
        self.spec_view.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        preview_box("Synthesis map (what you hear)", self.spec_view)

        # ---------------- parameter groups (2×2) ----------------
        pgrid = QGridLayout()
        pgrid.setSpacing(8)

        def group(title):
            g = QGroupBox(title)
            f = QFormLayout(g)
            f.setLabelAlignment(Qt.AlignRight)
            f.setHorizontalSpacing(8)
            f.setVerticalSpacing(3)
            return g, f

        def row(*widgets):
            w = QWidget()
            h = QHBoxLayout(w)
            h.setContentsMargins(0, 0, 0, 0)
            h.setSpacing(6)
            for x in widgets:
                h.addWidget(x)
            h.addStretch(1)
            return w

        DIVS = [("1/1 (per bar)", 4.0), ("1/2", 2.0), ("1/4 (per beat)", 1.0),
                ("1/8", 0.5), ("1/16", 0.25)]

        # ---- Sound ----
        gs, fs = group("Sound")

        self.mode_combo = QComboBox()
        for label, val in [("Oscillator bank (harmonics)", "osc"),
                           ("Waveform scan (wavetable)", "wave"),
                           ("Raster scan (fax)", "raster"),
                           ("Spectrogram (legacy)", "spec")]:
            self.mode_combo.addItem(label, val)
        fs.addRow("Synth mode", self.mode_combo)

        self.timbre_combo = QComboBox()
        for label, val in [("Sine", "sine"), ("Triangle", "triangle"),
                           ("Square", "square"), ("Saw", "saw")]:
            self.timbre_combo.addItem(label, val)
        self.osc = QComboBox()
        self.osc.addItems(["64", "128", "192", "256", "384", "512"])
        self.osc.setCurrentText("256")
        fs.addRow("Timbre / osc", row(self.timbre_combo, self.osc))

        self.chan_combo = QComboBox()
        for label, val in [("Mono (luminance)", "mono"),
                           ("RGB timbre banks (R=saw G=tri B=sine)", "rgb"),
                           ("Stereo R→left / B→right", "stereo")]:
            self.chan_combo.addItem(label, val)
        fs.addRow("Channels", self.chan_combo)

        self.fmin = QSpinBox(); self.fmin.setRange(20, 2000)
        self.fmin.setValue(50); self.fmin.setSuffix(" Hz")
        self.fmax = QSpinBox(); self.fmax.setRange(500, 22000)
        self.fmax.setValue(8000); self.fmax.setSuffix(" Hz")
        fs.addRow("F min / F max", row(self.fmin, self.fmax))

        self.scale = QComboBox()
        self.scale.addItems(["Logarithmic", "Linear"])
        self.gamma = QDoubleSpinBox(); self.gamma.setRange(0.2, 4.0)
        self.gamma.setSingleStep(0.1); self.gamma.setValue(1.4)
        fs.addRow("Scale / contrast γ", row(self.scale, self.gamma))

        self.fft = QComboBox(); self.fft.addItems(["1024", "2048", "4096"])
        self.fft.setCurrentText("2048")
        self.gl = QSpinBox(); self.gl.setRange(0, 60); self.gl.setValue(16)
        fs.addRow("FFT / phase (spec)", row(self.fft, self.gl))

        self.invert = QCheckBox("invert (dark = loud)")
        fs.addRow("", self.invert)
        pgrid.addWidget(gs, 0, 0)

        # ---- Timing ----
        gt, ft = group("Timing — tempo, sweep, envelope")

        self.bpm = QSpinBox(); self.bpm.setRange(40, 240)
        self.bpm.setValue(120); self.bpm.setSuffix(" bpm")
        self.beats = QSpinBox(); self.beats.setRange(1, 32); self.beats.setValue(16)
        ft.addRow("Tempo / beats", row(self.bpm, self.beats))
        self.len_label = QLabel()
        self.len_label.setStyleSheet("font-weight:700;")
        ft.addRow("Length", self.len_label)

        self.follow_len = QCheckBox("follow length")
        self.follow_len.setChecked(True)
        self.read_spin = QDoubleSpinBox(); self.read_spin.setRange(0.1, 120.0)
        self.read_spin.setSingleStep(0.5); self.read_spin.setValue(8.0)
        self.read_spin.setSuffix(" s")
        ft.addRow("Read time", row(self.follow_len, self.read_spin))

        self.repeats = QSpinBox(); self.repeats.setRange(1, 16); self.repeats.setValue(1)
        ft.addRow("Repeats (re-scans)", self.repeats)

        self.pluck = QCheckBox("pluck")
        self.pluck_div = QComboBox()
        for label, val in DIVS:
            self.pluck_div.addItem(label, val)
        self.pluck_div.setCurrentIndex(2)
        ft.addRow("Pluck env", row(self.pluck, self.pluck_div))

        self.attack_ms = QSpinBox(); self.attack_ms.setRange(1, 500)
        self.attack_ms.setValue(8); self.attack_ms.setSuffix(" ms")
        self.decay_ms = QSpinBox(); self.decay_ms.setRange(30, 4000)
        self.decay_ms.setValue(350); self.decay_ms.setSuffix(" ms")
        ft.addRow("Attack / decay", row(self.attack_ms, self.decay_ms))

        self.strum = QSpinBox(); self.strum.setRange(0, 100)
        self.strum.setValue(20); self.strum.setSuffix(" %")
        ft.addRow("Strum (row cascade)", self.strum)

        self.pulse = QCheckBox("gate")
        self.pulse_div = QComboBox()
        for label, val in DIVS:
            self.pulse_div.addItem(label, val)
        self.pulse_div.setCurrentIndex(2)
        ft.addRow("Pulse gate", row(self.pulse, self.pulse_div))

        self.pulse_depth = QSpinBox(); self.pulse_depth.setRange(0, 100)
        self.pulse_depth.setValue(70); self.pulse_depth.setSuffix(" %")
        ft.addRow("Pulse depth", self.pulse_depth)

        self.click = QCheckBox("metronome click")
        ft.addRow("Click", self.click)
        pgrid.addWidget(gt, 0, 1)

        # ---- Effects ----
        ge, fe = group("Effects")

        self.drive = QSpinBox(); self.drive.setRange(0, 100)
        self.drive.setValue(0); self.drive.setSuffix(" %")
        fe.addRow("Drive (distortion)", self.drive)

        self.cutoff = QSpinBox(); self.cutoff.setRange(100, 20000)
        self.cutoff.setValue(20000); self.cutoff.setSuffix(" Hz")
        self.reso = QDoubleSpinBox(); self.reso.setRange(0.5, 12.0)
        self.reso.setSingleStep(0.1); self.reso.setValue(0.7)
        fe.addRow("Cutoff / reso", row(self.cutoff, self.reso))

        self.delay_sync = QComboBox()
        for label, val in [("Off", 0.0), ("1/2", 2.0), ("1/4 dotted", 1.5),
                           ("1/4", 1.0), ("1/8 dotted", 0.75), ("1/8", 0.5),
                           ("1/16", 0.25)]:
            self.delay_sync.addItem(label, val)
        fe.addRow("Delay sync", self.delay_sync)

        self.delay_fb = QSpinBox(); self.delay_fb.setRange(0, 95)
        self.delay_fb.setValue(35); self.delay_fb.setSuffix(" %")
        self.delay_mix = QSpinBox(); self.delay_mix.setRange(0, 100)
        self.delay_mix.setValue(0); self.delay_mix.setSuffix(" %")
        fe.addRow("Delay fb / mix", row(self.delay_fb, self.delay_mix))

        self.rev_size = QSpinBox(); self.rev_size.setRange(10, 100)
        self.rev_size.setValue(50); self.rev_size.setSuffix(" %")
        self.rev_mix = QSpinBox(); self.rev_mix.setRange(0, 100)
        self.rev_mix.setValue(15); self.rev_mix.setSuffix(" %")
        fe.addRow("Reverb size / mix", row(self.rev_size, self.rev_mix))
        pgrid.addWidget(ge, 1, 0)

        # ---- Render / seed ----
        gr_, fr_ = group("Render — seed & selection")

        self.seed_spin = QSpinBox(); self.seed_spin.setRange(0, 2147483647)
        self.btn_dice = QPushButton("🎲")
        self.btn_dice.setToolTip("Re-roll seed")
        self.btn_dice.setFocusPolicy(Qt.TabFocus)
        self.btn_dice.setFixedWidth(36)
        fr_.addRow("Seed", row(self.seed_spin, self.btn_dice))

        self.seed_lock = QCheckBox("lock seed (reproducible renders)")
        fr_.addRow("", self.seed_lock)

        tip = QLabel("Select a thumbnail, drag a region on it if you like. "
                     "Read time = playhead speed; Repeats re-scan the image. "
                     "Double-click a thumbnail to Synthesize it instantly.")
        tip.setWordWrap(True)
        tip.setStyleSheet("color:#666;")
        fr_.addRow(tip)
        pgrid.addWidget(gr_, 1, 1)

        pgrid.setColumnStretch(0, 1)
        pgrid.setColumnStretch(1, 1)
        root.addLayout(pgrid)

        # wiring
        self.bpm.valueChanged.connect(self._update_len)
        self.beats.valueChanged.connect(self._update_len)
        self.follow_len.toggled.connect(self._on_follow_toggled)
        self.btn_dice.clicked.connect(self._reroll_seed)
        self.mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        self._update_len()
        self._on_mode_changed()

        # drag & drop overlay (NEW)
        self.drop_overlay = QLabel("⬇  Drop images or a folder here", central)
        self.drop_overlay.setAlignment(Qt.AlignCenter)
        self.drop_overlay.setStyleSheet("""
            background: rgba(30,110,255,140); color: white;
            border: 3px dashed #cfe0ff; border-radius: 12px;
            font-size: 22px; font-weight: 600;
        """)
        self.drop_overlay.hide()

    # ------------------------------------------- NEW: image management --
    def add_images_dialog(self):
        paths, _ = QFileDialog.getOpenFileNames(self, "Add images", "",
                                                IMAGE_FILTER)
        if paths:
            self.add_image_paths(paths)

    def add_image_paths(self, paths):
        added = 0
        for p in paths:
            p = os.path.normpath(p)
            if any(d["path"] == p for d in self.images):
                continue
            try:
                pil = load_pil(p)
            except Exception as exc:
                QMessageBox.warning(self, "Add image",
                                    f"Could not load:\n{p}\n\n{exc}")
                continue
            self._append_image({"path": p, "name": os.path.basename(p),
                                "pil": pil, "region": None})
            added += 1
        if added:
            self._select_row(self.gallery.count() - 1)
            self.statusBar().showMessage(f"Added {added} image(s) — "
                                         f"{len(self.images)} loaded.")
        self._update_count()

    def add_pil_image(self, pil, name):
        d = {"path": "", "name": name, "pil": pil, "region": None}
        self._append_image(d)
        self._select_row(self.gallery.count() - 1)
        self._update_count()
        self.statusBar().showMessage(f"Added {name} — {len(self.images)} "
                                     f"image(s) loaded.")

    def _append_image(self, d):
        self.images.append(d)
        pil = d["pil"]
        th = pil.convert("RGB").copy()
        th.thumbnail((THUMB.width() * 2, THUMB.height() * 2), RESAMPLE)
        buf = io.BytesIO()
        th.save(buf, format="PNG")
        pm = QPixmap()
        pm.loadFromData(buf.getvalue(), "PNG")
        icon = QIcon(pm.scaled(THUMB, Qt.KeepAspectRatio,
                               Qt.SmoothTransformation))
        label = d["name"]
        if len(label) > 18:
            label = label[:16] + "…"
        item = QListWidgetItem(icon, label)
        item.setData(Qt.UserRole, d)
        item.setToolTip(f"{d['name']}\n{pil.width()}×{pil.height()} px"
                        + (f"\n{d['path']}" if d["path"] else ""))
        self.gallery.blockSignals(True)
        self.gallery.addItem(item)
        self.gallery.blockSignals(False)

    def _select_row(self, row):
        self.gallery.setCurrentRow(row)

    def _current_data(self):
        it = self.gallery.currentItem()
        return it.data(Qt.UserRole) if it is not None else None

    def _on_gallery_select(self):
        d = self._current_data()
        if d is None:
            self.image = None
            self.image_name = ""
            self.image_view.set_image(None)
            self.btn_clear_region.setEnabled(False)
            self._update_count()
            return
        self.image = d["pil"]
        self.image_name = d["name"]
        self.image_view.set_image(self._pil_to_pixmap(d["pil"]), d["region"])
        self.btn_clear_region.setEnabled(d["region"] is not None)
        self._update_count()

    def remove_selected_image(self):
        row = self.gallery.currentRow()
        if row < 0:
            return
        self.gallery.blockSignals(True)
        self.gallery.takeItem(row)
        del self.images[row]
        self.gallery.blockSignals(False)
        self._on_gallery_select()          # refresh selection / previews
        self.statusBar().showMessage(f"Removed image — {len(self.images)} "
                                     f"remaining.")

    def clear_all_images(self):
        if not self.images:
            return
        self.gallery.blockSignals(True)
        self.gallery.clear()
        self.images.clear()
        self.gallery.blockSignals(False)
        self.image = None
        self.image_name = ""
        self.image_view.set_image(None)
        self.btn_clear_region.setEnabled(False)
        self._update_count()
        self.statusBar().showMessage("Cleared all images.")

    def _update_count(self):
        n = len(self.images)
        sel = f"  •  {self.image_name}" if self.image_name else ""
        self.gal_count.setText(f"{n} image{'s' if n != 1 else ''}{sel}")

    def _paste_clipboard(self):
        cb = QApplication.clipboard()
        pil = qimage_to_pil(cb.image())
        if pil is not None:
            self._paste_count += 1
            self.add_pil_image(pil, f"pasted_{self._paste_count}.png")
            return
        text = (cb.text() or "").strip()
        if text and os.path.isfile(text) and \
                text.lower().endswith(IMAGE_EXTS):
            self.add_image_paths([text])
        else:
            self.statusBar().showMessage(
                "Clipboard has no image (or image path).", 4000)

    # -------------------------------------------- NEW: drag & drop + overlay --
    def _collect_drop_paths(self, urls):
        paths = []
        for url in urls:
            p = url.toLocalFile()
            if not p:
                continue
            if os.path.isdir(p):
                for f in sorted(os.listdir(p)):
                    if f.lower().endswith(IMAGE_EXTS):
                        paths.append(os.path.join(p, f))
            elif p.lower().endswith(IMAGE_EXTS):
                paths.append(p)
        return paths

    def dragEnterEvent(self, event):
        md = event.mimeData()
        if md.hasUrls() or md.hasImage():
            event.acceptProposedAction()
            self.drop_overlay.setGeometry(self.centralWidget().rect())
            self.drop_overlay.raise_()
            self.drop_overlay.show()

    def dragLeaveEvent(self, event):
        self.drop_overlay.hide()

    def dropEvent(self, event):
        self.drop_overlay.hide()
        paths = self._collect_drop_paths(event.mimeData().urls()) \
            if event.mimeData().hasUrls() else []
        if paths:
            event.acceptProposedAction()
            self.add_image_paths(paths)
        elif event.mimeData().hasImage():
            pil = qimage_to_pil(event.mimeData().imageData())
            if pil is not None:
                event.acceptProposedAction()
                self._paste_count += 1
                self.add_pil_image(pil, f"dropped_{self._paste_count}.png")

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if not self.drop_overlay.isHidden():
            self.drop_overlay.setGeometry(self.centralWidget().rect())
        pm = getattr(self.spec_view, "_full", None)
        if pm:
            self.spec_view.setPixmap(pm.scaled(
                self.spec_view.size(), Qt.KeepAspectRatio,
                Qt.SmoothTransformation))

    # --------------------------------------------------------- shortcuts --
    def _make_shortcuts(self):
        QShortcut(QKeySequence(Qt.Key_Space), self,
                  activated=self.toggle_play_pause)
        QShortcut(QKeySequence(Qt.Key_S), self, activated=self.stop_playback)
        QShortcut(QKeySequence(Qt.Key_Home), self, activated=self.rewind)
        QShortcut(QKeySequence.StandardKey.Paste, self,
                  activated=self._paste_clipboard)
        QShortcut(QKeySequence("Ctrl+O"), self,
                  activated=self.add_images_dialog)

    # ------------------------------------------------- transport actions --
    def toggle_play_pause(self):
        if self._busy:
            return
        if self.wav_bytes is None:
            self.synthesize()
            return
        state = self.player.playbackState()
        if state == QMediaPlayer.PlaybackState.PlayingState:
            self.player.pause()
            self.statusBar().showMessage("Paused.")
        elif state == QMediaPlayer.PlaybackState.PausedState:
            self.player.play()
        else:
            if (self.player.duration() > 0
                    and self.player.position() >= self.player.duration()):
                self.player.setPosition(0)
            self.play()

    def stop_playback(self):
        self.player.stop()
        self.player.setPosition(0)
        self.seek.setValue(0)
        self.statusBar().showMessage("Stopped.")

    def rewind(self):
        was_playing = (self.player.playbackState()
                       == QMediaPlayer.PlaybackState.PlayingState)
        self.player.setPosition(0)
        self.seek.setValue(0)
        if was_playing:
            self.player.play()

    def _seek_pressed(self):
        self._scrubbing = True

    def _seek_released(self):
        self._scrubbing = False
        self.player.setPosition(self.seek.value())

    def _on_duration(self, ms):
        if ms > 0:
            self.seek.setRange(0, int(ms))

    # --------------------------------------------------------- previews --
    @staticmethod
    def _pil_to_pixmap(img):
        buf = io.BytesIO()
        img.convert("RGB").save(buf, format="PNG")
        pm = QPixmap()
        pm.loadFromData(buf.getvalue(), "PNG")
        return pm

    # ------------------------------------------- UI helper slots --
    def _update_len(self):
        track = self.beats.value() * 60.0 / self.bpm.value()
        self.len_label.setText(f"{track:.2f} s  ({self.beats.value() / 4.0:g} bars)")
        if self.follow_len.isChecked():
            self.read_spin.setValue(round(track, 2))

    def _on_follow_toggled(self, on):
        self.read_spin.setEnabled(not on
                                  and self.mode_combo.currentData() != "spec")
        if on:
            self._update_len()

    def _reroll_seed(self):
        self.seed_spin.setValue(
            int(np.random.default_rng().integers(0, 2 ** 31)))

    def _on_mode_changed(self, *_):
        mode = self.mode_combo.currentData()
        is_osc, is_spec = mode == "osc", mode == "spec"
        uses_freq = is_osc or is_spec
        self.timbre_combo.setEnabled(is_osc)
        self.osc.setEnabled(is_osc)
        self.fft.setEnabled(is_spec)
        self.gl.setEnabled(is_spec)
        self.fmin.setEnabled(uses_freq)
        self.fmax.setEnabled(uses_freq)
        self.scale.setEnabled(uses_freq)
        self.chan_combo.setEnabled(not is_spec)
        self.follow_len.setEnabled(not is_spec)
        self.read_spin.setEnabled(not is_spec and not self.follow_len.isChecked())
        self.repeats.setEnabled(not is_spec)
        self.strum.setEnabled(is_osc)

    def _on_region(self, rect):
        d = self._current_data()
        if d is not None:
            d["region"] = rect          # remember per image
        self.btn_clear_region.setEnabled(rect is not None)
        if rect is not None:
            self.statusBar().showMessage(
                f"Region {rect.width()}×{rect.height()} px at "
                f"({rect.x()}, {rect.y()}) — Synthesize to sonify it.")
        else:
            self.statusBar().showMessage("Region cleared — full image.")

    def _clear_region(self):
        self.image_view.clear_region()

    # -------------------------------------------------------- synthesis --
    def _collect_params(self):
        track = self.beats.value() * 60.0 / self.bpm.value()
        read_s = track if self.follow_len.isChecked() else self.read_spin.value()
        return dict(
            mode=self.mode_combo.currentData(),
            timbre=self.timbre_combo.currentData(),
            channels=self.chan_combo.currentData(),
            bpm=self.bpm.value(), beats=self.beats.value(),
            read_s=float(read_s), repeats=self.repeats.value(),
            pluck=self.pluck.isChecked(),
            pluck_div=self.pluck_div.currentData(),
            attack=self.attack_ms.value() / 1000.0,
            decay=self.decay_ms.value() / 1000.0,
            strum=self.strum.value() / 100.0,
            pulse=self.pulse.isChecked(),
            pulse_div=self.pulse_div.currentData(),
            pulse_depth=self.pulse_depth.value() / 100.0,
            click=self.click.isChecked(),
            fmin=self.fmin.value(), fmax=self.fmax.value(),
            log_scale=self.scale.currentIndex() == 0,
            gamma=self.gamma.value(), invert=self.invert.isChecked(),
            gl_iters=self.gl.value(), osc_rows=int(self.osc.currentText()),
            drive=self.drive.value() / 100.0,
            cutoff=float(self.cutoff.value()), reso=self.reso.value(),
            delay_div=self.delay_sync.currentData(),
            delay_fb=self.delay_fb.value() / 100.0,
            delay_mix=self.delay_mix.value() / 100.0,
            rev_size=self.rev_size.value() / 100.0,
            rev_mix=self.rev_mix.value() / 100.0,
            seed=self.seed_spin.value() if self.seed_lock.isChecked() else None)

    def synthesize(self):
        if self._busy:
            return
        if self.image is None:
            self.add_images_dialog()
            if self.image is None:
                return
        img = self.image
        reg = self.image_view.image_region()
        if reg is not None:
            img = img.crop((reg.x(), reg.y(),
                            reg.x() + reg.width(), reg.y() + reg.height()))

        n_fft = int(self.fft.currentText())
        self.synth.n_fft = n_fft
        self.synth.hop = n_fft // 4

        params = self._collect_params()
        self.mode = params["mode"]
        read, reps = params["read_s"], params["repeats"]
        track = params["beats"] * 60.0 / params["bpm"]
        self.scan_windows = [(k * read, min((k + 1) * read, track))
                             for k in range(reps) if k * read < track - 1e-6]

        self._set_busy(True, f"Synthesizing “{self.image_name}” "
                             f"({self.mode_combo.currentText()})… "
                             f"{params['beats']} beats @ {params['bpm']} BPM, "
                             f"read {read:.2f} s × {reps}")
        self.worker = SynthWorker(self.synth, img, params)
        self.worker.finished_ok.connect(self._on_synth_done)
        self.worker.failed.connect(self._on_synth_failed)
        self.worker.start()

    def _set_busy(self, busy, message=""):
        self._busy = busy
        for b in (self.btn_synth, self.btn_add, self.btn_save, self.btn_mp4):
            b.setEnabled(not busy)
        if busy:
            QApplication.setOverrideCursor(Qt.WaitCursor)
        else:
            QApplication.restoreOverrideCursor()
        if message:
            self.statusBar().showMessage(message)

    def _on_synth_done(self, audio, audio_map, seed):
        self.audio = audio
        self.map2d = audio_map
        self.duration = audio.shape[0] / self.synth.sample_rate
        self.wav_bytes = audio_to_wav_bytes(audio, self.synth.sample_rate)
        self.seed_spin.setValue(seed)

        if self.mode == "spec":
            pixmap = spectrogram_pixmap(audio_map)
        else:
            pixmap = canvas_pixmap(audio_map)
        self.spec_view._full = pixmap
        self.spec_view.setPixmap(pixmap.scaled(
            self.spec_view.size(), Qt.KeepAspectRatio,
            Qt.SmoothTransformation))

        total_ms = int(self.duration * 1000)
        self.seek.setRange(0, total_ms)
        self.seek.setEnabled(True)
        self.seek.setValue(0)
        self.time_label.setText(f"0:00.0 / {_fmt_time(total_ms)}")

        locked = "locked" if self.seed_lock.isChecked() else "lock to reuse"
        self._set_busy(False, f"Done — seed {seed} ({locked}) — playing…")
        self.play()

    def _on_synth_failed(self, message):
        self._set_busy(False)
        QMessageBox.critical(self, "Synthesis failed", message)

    # --------------------------------------------------------- MP4 export --
    def open_export(self):
        if self.wav_bytes is None or self.image is None or self.map2d is None:
            QMessageBox.information(
                self, "Export MP4",
                "Synthesize first: select an image and press Synthesize.")
            return
        exe = find_ffmpeg()
        if exe is None:
            QMessageBox.warning(
                self, "Export MP4",
                "ffmpeg was not found.\n\nInstall the bundled version with:\n"
                "    pip install imageio-ffmpeg")
            return
        ExportDialog(self, exe).exec()

    # ---------------------------------------------------------- playback --
    def play(self):
        if self.wav_bytes is None:
            self.synthesize()
            return
        self.player.stop()
        buf = QBuffer(self)
        buf.setData(QByteArray(self.wav_bytes))
        buf.open(QBuffer.ReadOnly)
        self.buffer = buf
        self.player.setSourceDevice(buf)
        self.player.play()

    def _on_playback_state(self, state):
        self.btn_play.setText("⏸ Pause"
                              if state == QMediaPlayer.PlaybackState.PlayingState
                              else "▶ Play")
        has = self.wav_bytes is not None
        self.btn_rewind.setEnabled(has)
        self.btn_stop.setEnabled(has)

    def _on_position(self, ms):
        if not self._scrubbing and self.player.duration() > 0:
            self.seek.blockSignals(True)
            self.seek.setValue(int(ms))
            self.seek.blockSignals(False)
        total = self.player.duration() or int(self.duration * 1000)
        self.time_label.setText(f"{_fmt_time(ms)} / {_fmt_time(total)}")

    def _on_media_status(self, status):
        if status == QMediaPlayer.MediaStatus.EndOfMedia:
            if self.loop.isChecked():
                self.player.setPosition(0)
                self.player.play()
            else:
                self.statusBar().showMessage("Finished. Press Space to replay.")

    def save_wav(self):
        if self.wav_bytes is None:
            QMessageBox.information(self, "Save WAV", "Synthesize something first.")
            return
        stem = os.path.splitext(self.image_name or "image_sound")[0]
        path, _ = QFileDialog.getSaveFileName(self, "Save WAV",
                                              f"{stem}.wav",
                                              "WAV audio (*.wav)")
        if not path:
            return
        if not path.lower().endswith(".wav"):
            path += ".wav"
        with open(path, "wb") as fh:
            fh.write(self.wav_bytes)
        self.statusBar().showMessage(f"Saved {path}")

    def closeEvent(self, event):
        self.player.stop()
        if self.worker is not None and self.worker.isRunning():
            self.worker.wait(3000)
        super().closeEvent(event)


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setStyleSheet("""
        QGroupBox { font-weight: 600; border: 1px solid #bbb;
                    border-radius: 6px; margin-top: 10px; }
        QGroupBox::title { subcontrol-origin: margin; left: 8px; padding: 0 4px; }
    """)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()