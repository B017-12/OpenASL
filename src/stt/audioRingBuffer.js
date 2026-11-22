/**
 * Pure, DOM-free 16kHz audio downsampler & sliding-window ring buffer with adaptive VAD.
 * Used inside whisper.worker.js and directly unit-tested under node:test.
 */

export const TARGET_SAMPLE_RATE = 16000;

export function downsampleTo16k(float32Input, inputSampleRate) {
  if (!float32Input || float32Input.length === 0) return new Float32Array(0);
  if (!inputSampleRate || inputSampleRate === TARGET_SAMPLE_RATE) {
    return float32Input instanceof Float32Array
      ? new Float32Array(float32Input)
      : Float32Array.from(float32Input);
  }
  const ratio = inputSampleRate / TARGET_SAMPLE_RATE;
  const newLen = Math.max(1, Math.round(float32Input.length / ratio));
  const out = new Float32Array(newLen);
  for (let i = 0; i < newLen; i++) {
    const start = Math.floor(i * ratio);
    const end = Math.min(float32Input.length, Math.max(start + 1, Math.floor((i + 1) * ratio)));
    let sum = 0;
    for (let j = start; j < end; j++) {
      sum += float32Input[j];
    }
    out[i] = sum / Math.max(1, end - start);
  }
  return out;
}

export class StreamingAudioBuffer {
  constructor(options = {}) {
    this.sampleRate = options.sampleRate || TARGET_SAMPLE_RATE;
    this.baseVadThreshold = options.baseVadThreshold ?? 0.012;
    this.noiseMultiplier = options.noiseMultiplier ?? 2.4;
    this.preRollMs = options.preRollMs ?? 180;
    this.maxWindowSec = options.maxWindowSec ?? 4.5;
    this.compactKeepSec = options.compactKeepSec ?? 1.4;
    this.minActiveSpeechMs = options.minActiveSpeechMs ?? 200;

    this.noiseFloor =
      typeof options.initialNoiseFloor === "number" &&
      Number.isFinite(options.initialNoiseFloor) &&
      options.initialNoiseFloor >= 0
        ? options.initialNoiseFloor
        : 0.005;
    this._preRollChunks = [];
    this._preRollSamples = 0;
    this.resetUtterance();
  }

  get preRollLimitSamples() {
    return Math.round((this.preRollMs / 1000) * this.sampleRate);
  }

  get activeSpeechMs() {
    return (this.activeSpeechSamples / this.sampleRate) * 1000;
  }

  get durationSec() {
    return this._utteranceSamples / this.sampleRate;
  }

  get threshold() {
    return Math.max(this.baseVadThreshold, this.noiseFloor * this.noiseMultiplier);
  }

  resetUtterance() {
    this._utteranceChunks = [];
    this._utteranceSamples = 0;
    this.isSpeaking = false;
    this.speechOnsetMs = 0;
    this.windowStartMs = 0;
    this.windowEndMs = 0;
    this.lastSpeechMs = 0;
    this.silenceMs = 0;
    this.activeSpeechSamples = 0;
    this.newSamplesSinceLastDecode = 0;
    this.peakRms = 0;
  }

  resetAll() {
    this.noiseFloor = 0.005;
    this._preRollChunks = [];
    this._preRollSamples = 0;
    this.resetUtterance();
  }

  _pushPreRoll(entry) {
    this._preRollChunks.push(entry);
    this._preRollSamples += entry.pcm.length;
    const limit = this.preRollLimitSamples;
    while (this._preRollChunks.length > 1 && this._preRollSamples - this._preRollChunks[0].pcm.length >= limit) {
      const removed = this._preRollChunks.shift();
      this._preRollSamples -= removed.pcm.length;
    }
  }

  pushChunk(pcm16k, rms, captureTimestampMs) {
    if (!pcm16k || pcm16k.length === 0) return { speechActive: this.isSpeaking };

    const safeRms = typeof rms === "number" && Number.isFinite(rms) && rms >= 0 ? rms : 0;
    const chunkMs = (pcm16k.length / this.sampleRate) * 1000;
    const endMs = Number.isFinite(captureTimestampMs) ? captureTimestampMs : this.windowEndMs + chunkMs;
    const startMs = Math.max(0, endMs - chunkMs);
    const entry = { pcm: pcm16k, startMs, endMs };

    const thresh = this.threshold;

    if (safeRms >= thresh) {
      if (!this.isSpeaking) {
        this.isSpeaking = true;
        this.speechOnsetMs = startMs;
        this.activeSpeechSamples = 0;
        this.silenceMs = 0;
        this.peakRms = safeRms;

        // Prepend bounded pre-roll audio so initial plosives aren't clipped
        this._utteranceChunks = [];
        this._utteranceSamples = 0;
        if (this._preRollChunks.length > 0) {
          for (const pre of this._preRollChunks) {
            this._utteranceChunks.push(pre);
            this._utteranceSamples += pre.pcm.length;
          }
          this.windowStartMs = this._utteranceChunks[0].startMs;
        } else {
          this.windowStartMs = startMs;
        }
        this.newSamplesSinceLastDecode = this._utteranceSamples;
      }

      this._utteranceChunks.push(entry);
      this._utteranceSamples += pcm16k.length;
      this.windowEndMs = endMs;
      this.lastSpeechMs = endMs;
      this.silenceMs = 0;
      this.activeSpeechSamples += pcm16k.length;
      this.newSamplesSinceLastDecode += pcm16k.length;
      if (safeRms > this.peakRms) this.peakRms = safeRms;
    } else {
      // Gently adapt ambient noise floor during silence
      this.noiseFloor = this.noiseFloor * 0.96 + safeRms * 0.04;
      this._pushPreRoll(entry);

      if (this.isSpeaking) {
        this._utteranceChunks.push(entry);
        this._utteranceSamples += pcm16k.length;
        this.windowEndMs = endMs;
        this.newSamplesSinceLastDecode += pcm16k.length;
        this.silenceMs = Math.max(0, endMs - this.lastSpeechMs);
      }
    }

    return { speechActive: this.isSpeaking, silenceMs: this.silenceMs };
  }

  shouldFinalize(microPauseMs = 280) {
    return (
      this.isSpeaking &&
      this.silenceMs >= microPauseMs &&
      this.activeSpeechMs >= this.minActiveSpeechMs
    );
  }

  isShortNoiseBurst(microPauseMs = 280) {
    return (
      this.isSpeaking &&
      this.silenceMs >= microPauseMs &&
      this.activeSpeechMs < this.minActiveSpeechMs
    );
  }

  markDecoded() {
    this.newSamplesSinceLastDecode = 0;
  }

  getWindowSlice() {
    if (this._utteranceSamples === 0) {
      return {
        audio: new Float32Array(0),
        windowStartMs: this.windowStartMs,
        windowEndMs: this.windowEndMs,
        durationSec: 0,
        peakRms: this.peakRms,
        activeSpeechMs: this.activeSpeechMs,
      };
    }

    const maxSamples = Math.round(this.maxWindowSec * this.sampleRate);
    const total = this._utteranceSamples;
    const merged = new Float32Array(total);
    let offset = 0;
    for (const c of this._utteranceChunks) {
      merged.set(c.pcm, offset);
      offset += c.pcm.length;
    }

    if (total <= maxSamples) {
      return {
        audio: merged,
        windowStartMs: this.windowStartMs,
        windowEndMs: this.windowEndMs,
        durationSec: total / this.sampleRate,
        peakRms: this.peakRms,
        activeSpeechMs: this.activeSpeechMs,
      };
    }

    const trimSamples = total - maxSamples;
    const trimmedAudio = merged.slice(trimSamples);
    const trimmedMs = (trimSamples / this.sampleRate) * 1000;
    return {
      audio: trimmedAudio,
      windowStartMs: this.windowStartMs + trimmedMs,
      windowEndMs: this.windowEndMs,
      durationSec: trimmedAudio.length / this.sampleRate,
      peakRms: this.peakRms,
      activeSpeechMs: this.activeSpeechMs,
    };
  }

  /**
   * Slides the rolling window left during long continuous speech, keeping `keepSec`
   * of overlap tail so boundary words are never chopped.
   */
  compactLeft(keepSec = this.compactKeepSec) {
    const keepSamples = Math.round(keepSec * this.sampleRate);
    if (this._utteranceSamples <= keepSamples) return false;

    const merged = new Float32Array(this._utteranceSamples);
    let offset = 0;
    for (const c of this._utteranceChunks) {
      merged.set(c.pcm, offset);
      offset += c.pcm.length;
    }

    const dropSamples = this._utteranceSamples - keepSamples;
    const keptPcm = merged.slice(dropSamples);
    const droppedMs = (dropSamples / this.sampleRate) * 1000;
    const newStartMs = this.windowStartMs + droppedMs;

    this._utteranceChunks = [{ pcm: keptPcm, startMs: newStartMs, endMs: this.windowEndMs }];
    this._utteranceSamples = keptPcm.length;
    this.windowStartMs = newStartMs;
    return true;
  }
}
