/**
 * Browser PCM Audio Capture using an inline-Blob AudioWorkletNode (2048-sample / ~42.6ms 16kHz
 * mono PCM + RMS computed on the audio rendering thread, zero-copy Transferable), with a
 * transparent muted 2048-sample ScriptProcessorNode fallback for Windows/restricted sandboxes.
 */

import { TARGET_SAMPLE_RATE, downsampleTo16k } from "./audioRingBuffer.js";

export const DEFAULT_PCM_BUFFER_SIZE = 2048;

const WORKLET_PROCESSOR_NAME = "openasl-pcm-capture-processor";

const WORKLET_SOURCE = `
class OpenASLPcmCaptureProcessor extends AudioWorkletProcessor {
  constructor(options) {
    super();
    const procOpts = (options && options.processorOptions) || {};
    this.targetRate = 16000;
    this.inputRate = sampleRate || 48000;
    const explicitBufferSize = Number(procOpts.bufferSize);
    if (Number.isFinite(explicitBufferSize) && explicitBufferSize >= 256) {
      this.batchSize = Math.round(explicitBufferSize);
    } else {
      const targetChunkMs = Number(procOpts.chunkMs) || 42.67;
      this.batchSize = Math.max(256, Math.round((this.inputRate * targetChunkMs) / 1000));
    }
    this.buffer = new Float32Array(this.batchSize);
    this.offset = 0;
  }

  downsample(input) {
    if (this.inputRate === this.targetRate) {
      return new Float32Array(input);
    }
    const ratio = this.inputRate / this.targetRate;
    const newLen = Math.max(1, Math.round(input.length / ratio));
    const out = new Float32Array(newLen);
    for (let i = 0; i < newLen; i++) {
      const start = Math.floor(i * ratio);
      const end = Math.min(input.length, Math.max(start + 1, Math.floor((i + 1) * ratio)));
      let sum = 0;
      for (let j = start; j < end; j++) sum += input[j];
      out[i] = sum / Math.max(1, end - start);
    }
    return out;
  }

  process(inputs) {
    const channel = inputs && inputs[0] && inputs[0][0];
    if (!channel || channel.length === 0) return true;

    let readPos = 0;
    while (readPos < channel.length) {
      const space = this.batchSize - this.offset;
      const toCopy = Math.min(space, channel.length - readPos);
      this.buffer.set(channel.subarray(readPos, readPos + toCopy), this.offset);
      this.offset += toCopy;
      readPos += toCopy;

      if (this.offset >= this.batchSize) {
        let sumSq = 0;
        for (let i = 0; i < this.batchSize; i++) {
          const s = this.buffer[i];
          if (Number.isFinite(s)) {
            sumSq += s * s;
          } else {
            this.buffer[i] = 0;
          }
        }
        const rawRms = Math.sqrt(sumSq / this.batchSize);
        const rms = Number.isFinite(rawRms) && rawRms >= 0 ? rawRms : 0;
        const pcm16k = this.downsample(this.buffer);
        this.port.postMessage({ pcm: pcm16k, rms }, [pcm16k.buffer]);
        this.offset = 0;
      }
    }
    return true;
  }
}

registerProcessor("${WORKLET_PROCESSOR_NAME}", OpenASLPcmCaptureProcessor);
`;

export async function createPcmCapture({
  audioCtx,
  mediaStream,
  chunkMs = 42.67,
  bufferSize = DEFAULT_PCM_BUFFER_SIZE,
  useWorklet = false,
  onChunk,
}) {
  const source = audioCtx.createMediaStreamSource(mediaStream);
  const resolvedBufferSize =
    Number.isFinite(bufferSize) && bufferSize >= 256 ? Math.round(bufferSize) : DEFAULT_PCM_BUFFER_SIZE;
  let active = true;
  let mode = "worklet";
  let workletNode = null;
  let processorNode = null;
  let muteGain = null;

  const canUseWorklet =
    useWorklet === true &&
    typeof AudioWorkletNode !== "undefined" &&
    audioCtx.audioWorklet &&
    typeof audioCtx.audioWorklet.addModule === "function" &&
    typeof Blob !== "undefined" &&
    typeof URL !== "undefined" &&
    typeof URL.createObjectURL === "function";

  if (canUseWorklet) {
    let blobUrl = null;
    try {
      const blob = new Blob([WORKLET_SOURCE], { type: "application/javascript" });
      blobUrl = URL.createObjectURL(blob);
      await audioCtx.audioWorklet.addModule(blobUrl);
      URL.revokeObjectURL(blobUrl);
      blobUrl = null;

      workletNode = new AudioWorkletNode(audioCtx, WORKLET_PROCESSOR_NAME, {
        numberOfInputs: 1,
        numberOfOutputs: 0,
        channelCount: 1,
        processorOptions: { chunkMs, bufferSize: resolvedBufferSize },
      });

      workletNode.port.onmessage = (event) => {
        if (!active) return;
        const data = event.data;
        if (!data || !data.pcm) return;
        const safeRms = Number.isFinite(data.rms) && data.rms >= 0 ? data.rms : 0;
        onChunk({
          pcm: data.pcm,
          rms: safeRms,
          captureTimestampMs: performance.now(),
        });
      };

      source.connect(workletNode);
    } catch {
      if (blobUrl) {
        try {
          URL.revokeObjectURL(blobUrl);
        } catch {
          // Ignore
        }
      }
      workletNode = null;
      mode = "scriptprocessor";
    }
  } else {
    mode = "scriptprocessor";
  }

  if (!workletNode) {
    mode = "scriptprocessor";
    const sampleRate = audioCtx.sampleRate || 48000;
    processorNode = audioCtx.createScriptProcessor(resolvedBufferSize, 1, 1);
    muteGain = audioCtx.createGain();
    muteGain.gain.value = 0;

    processorNode.onaudioprocess = (e) => {
      if (!active) return;
      const input = e.inputBuffer.getChannelData(0);
      const clean = new Float32Array(input.length);
      let sumSq = 0;
      for (let i = 0; i < input.length; i++) {
        const s = input[i];
        if (Number.isFinite(s)) {
          clean[i] = s;
          sumSq += s * s;
        }
      }
      const rawRms = input.length > 0 ? Math.sqrt(sumSq / input.length) : 0;
      const rms = Number.isFinite(rawRms) && rawRms >= 0 ? rawRms : 0;
      const pcm = downsampleTo16k(clean, sampleRate);
      onChunk({
        pcm,
        rms,
        captureTimestampMs: performance.now(),
      });
    };

    source.connect(processorNode);
    processorNode.connect(muteGain);
    muteGain.connect(audioCtx.destination);
  }

  return {
    mode,
    bufferSize: resolvedBufferSize,
    targetSampleRate: TARGET_SAMPLE_RATE,
    stop() {
      active = false;
      try {
        source.disconnect();
      } catch {
        // Ignore
      }
      if (workletNode) {
        try {
          workletNode.port.onmessage = null;
          workletNode.disconnect();
        } catch {
          // Ignore
        }
        workletNode = null;
      }
      if (processorNode) {
        try {
          processorNode.onaudioprocess = null;
          processorNode.disconnect();
        } catch {
          // Ignore
        }
        processorNode = null;
      }
      if (muteGain) {
        try {
          muteGain.disconnect();
        } catch {
          // Ignore
        }
        muteGain = null;
      }
    },
  };
}
