/**
 * Loads public/assets/manifest.json, preloads A-Z fingerspelling SVGs,
 * trims stationary lead-in/lead-out frames on WLASL poses at load time so signing
 * begins immediately on Frame 0, and pre-warms all 257 poses in idle batches.
 */

const KEY_HAND_INDICES = [0, 4, 8, 12, 16, 20];
const KEY_ARM_INDICES = [3, 4, 5, 6];

function loadImage(src) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = () => reject(new Error(`Failed to load image: ${src}`));
    img.src = src;
  });
}

function measureFrameDelta(f0, f1) {
  if (!f0 || !f1) return 0;
  let total = 0;
  let count = 0;

  for (const key of ["lh", "rh"]) {
    const a0 = f0[key];
    const a1 = f1[key];
    if (Array.isArray(a0) && Array.isArray(a1)) {
      for (const idx of KEY_HAND_INDICES) {
        const p0 = a0[idx];
        const p1 = a1[idx];
        if (p0 && p1) {
          total += Math.hypot(p1[0] - p0[0], p1[1] - p0[1]);
          count++;
        }
      }
    }
  }

  const p0Arr = f0.p;
  const p1Arr = f1.p;
  if (Array.isArray(p0Arr) && Array.isArray(p1Arr)) {
    for (const idx of KEY_ARM_INDICES) {
      const p0 = p0Arr[idx];
      const p1 = p1Arr[idx];
      if (p0 && p1) {
        total += Math.hypot(p1[0] - p0[0], p1[1] - p0[1]);
        count++;
      }
    }
  }

  return count > 0 ? total / count : 0;
}

function isRestingFrame(frame) {
  if (!frame) return true;
  const hasHands =
    (Array.isArray(frame.lh) && frame.lh.some(Boolean)) ||
    (Array.isArray(frame.rh) && frame.rh.some(Boolean));
  if (hasHands) return false;

  const p = frame.p;
  if (!Array.isArray(p)) return true;
  const lw = p[5];
  const rw = p[6];
  const lwResting = !lw || lw[1] > 0.82;
  const rwResting = !rw || rw[1] > 0.82;
  return lwResting && rwResting;
}

/**
 * Trims dead stationary lead-in and lead-out frames from MediaPipe WLASL pose sequences
 * at load time (~120–200ms faster visual sign onset on Frame 0).
 * Short synthetic/test clips (<= 8 frames) are returned untouched.
 */
export function trimStationaryPoseFrames(frames) {
  if (!Array.isArray(frames) || frames.length <= 8) {
    return frames;
  }

  const n = frames.length;
  const deltas = new Float32Array(n - 1);
  let peakDelta = 0;
  for (let i = 0; i < n - 1; i++) {
    const d = measureFrameDelta(frames[i], frames[i + 1]);
    deltas[i] = d;
    if (d > peakDelta) peakDelta = d;
  }

  const motionThreshold = Math.max(0.0035, peakDelta * 0.14);
  const maxLeadTrim = Math.min(8, Math.floor(n * 0.28));
  const maxTailTrim = Math.min(8, Math.floor(n * 0.28));

  let start = 0;
  while (start < maxLeadTrim && (deltas[start] < motionThreshold || isRestingFrame(frames[start]))) {
    start++;
  }

  const minEnd = Math.max(start + 9, n - 1 - maxTailTrim);
  let end = n - 1;
  while (end > minEnd && (deltas[end - 1] < motionThreshold || isRestingFrame(frames[end]))) {
    end--;
  }

  if (start === 0 && end === n - 1) {
    return frames;
  }
  return frames.slice(start, end + 1);
}

export class ASLAssetStore {
  constructor() {
    this.manifest = { letters: {}, poses: {}, aliases: {} };
    this.letterImages = new Map();
    this.poseCache = new Map();
    this.inFlightPoses = new Map();
    this.localPoseOverrides = new Map();
    this._idlePrewarmHandle = null;
    this._idlePrewarmCancelled = false;
  }

  async init() {
    const resp = await fetch("/assets/manifest.json", { cache: "no-store" });
    if (!resp.ok) {
      throw new Error(`Could not load /assets/manifest.json (${resp.status})`);
    }
    this.manifest = await resp.json();

    const entries = Object.entries(this.manifest.letters || {});
    await Promise.all(
      entries.map(async ([letter, src]) => {
        try {
          const img = await loadImage(src);
          this.letterImages.set(letter, img);
        } catch (err) {
          console.warn(err.message);
        }
      })
    );

    return {
      letterCount: this.letterImages.size,
      poseCount: Object.keys(this.manifest.poses || {}).length,
      aliasCount: Object.keys(this.manifest.aliases || {}).length,
    };
  }

  getLetterImage(letter) {
    if (typeof letter !== "string" || !letter) return null;
    return this.letterImages.get(letter.toUpperCase()) || null;
  }

  resolvePoseEntry(gloss) {
    if (typeof gloss !== "string") return null;
    const key = gloss.toUpperCase().trim();
    if (!key) return null;

    if (this.localPoseOverrides.has(key)) {
      return { canonical: key, data: this.localPoseOverrides.get(key), isOverride: true };
    }
    if (this.manifest.poses?.[key]) {
      return { canonical: key, src: this.manifest.poses[key], isOverride: false };
    }
    const aliased = this.manifest.aliases?.[key];
    if (aliased && this.localPoseOverrides.has(aliased)) {
      return { canonical: aliased, data: this.localPoseOverrides.get(aliased), isOverride: true };
    }
    if (aliased && this.manifest.poses?.[aliased]) {
      return { canonical: aliased, src: this.manifest.poses[aliased], isOverride: false };
    }
    return null;
  }

  /**
   * Synchronous memory-cache lookup so `ASLStagePlayer.appendTokens` can bypass async
   * microtasks completely when poses are already warm in RAM.
   */
  getCachedPoseData(entry) {
    if (!entry) return null;
    if (entry.data) return entry.data;
    if (entry.src && this.poseCache.has(entry.src)) {
      return this.poseCache.get(entry.src);
    }
    return null;
  }

  async loadPoseData(entry) {
    if (!entry) return null;
    if (entry.data) return entry.data;
    if (this.poseCache.has(entry.src)) {
      return this.poseCache.get(entry.src);
    }
    if (this.inFlightPoses.has(entry.src)) {
      return this.inFlightPoses.get(entry.src);
    }

    const promise = (async () => {
      const resp = await fetch(entry.src);
      if (!resp.ok) {
        throw new Error(`Failed to load pose JSON: ${entry.src}`);
      }
      const raw = await resp.json();
      if (!Array.isArray(raw.frames) || raw.frames.length === 0) {
        throw new Error(`Pose JSON has no frames: ${entry.src}`);
      }
      const trimmedFrames = trimStationaryPoseFrames(raw.frames);
      const data = trimmedFrames === raw.frames ? raw : { ...raw, frames: trimmedFrames };
      this.poseCache.set(entry.src, data);
      return data;
    })();

    this.inFlightPoses.set(entry.src, promise);
    try {
      return await promise;
    } finally {
      this.inFlightPoses.delete(entry.src);
    }
  }

  async registerLocalPoseFile(file, overrideGloss = "") {
    const text = await file.text();
    const parsed = JSON.parse(text);
    if (!Array.isArray(parsed.frames) || parsed.frames.length === 0) {
      throw new Error("Invalid pose JSON: expected a non-empty 'frames' array.");
    }

    const key = (overrideGloss || parsed.gloss || file.name.replace(/\.json$/i, ""))
      .toUpperCase()
      .trim()
      .replace(/[\s-]+/g, "_");

    if (!key) throw new Error("Could not determine a gloss name for this pose JSON.");

    const trimmedFrames = trimStationaryPoseFrames(parsed.frames);
    this.localPoseOverrides.set(key, {
      ...parsed,
      frames: trimmedFrames,
      gloss: key,
      fileName: file.name,
    });
    return { gloss: key, frameCount: trimmedFrames.length, fileName: file.name };
  }

  clearLocalPose(gloss) {
    this.localPoseOverrides.delete(gloss.toUpperCase().trim());
  }

  listLocalPoses() {
    return Array.from(this.localPoseOverrides.entries()).map(([gloss, data]) => ({
      gloss,
      fileName: data.fileName || `${gloss}.json`,
      frameCount: data.frames?.length || 0,
    }));
  }

  /**
   * Returns a Set of all gloss keys that resolve to a real pose (local overrides,
   * manifest poses, and aliases whose target exists in manifest poses or overrides).
   */
  getKnownGlosses() {
    const known = new Set();
    for (const key of this.localPoseOverrides.keys()) {
      known.add(key);
    }
    const poses = this.manifest?.poses || {};
    for (const key of Object.keys(poses)) {
      known.add(key);
    }
    const aliases = this.manifest?.aliases || {};
    for (const [alias, target] of Object.entries(aliases)) {
      if (poses[target] || this.localPoseOverrides.has(target)) {
        known.add(alias);
      }
    }
    return known;
  }

  /**
   * Warm-caches high-frequency pose JSONs asynchronously in the background so first
   * spoken conversational signs hit memory cache with 0ms fetch latency.
   */
  async prefetchPoses(glosses = []) {
    if (!Array.isArray(glosses) || glosses.length === 0) return 0;
    let loaded = 0;
    await Promise.all(
      glosses.map(async (gloss) => {
        const entry = this.resolvePoseEntry(gloss);
        if (!entry || entry.data || this.poseCache.has(entry.src)) return;
        try {
          await this.loadPoseData(entry);
          loaded++;
        } catch {
          // Non-fatal background prefetch miss
        }
      })
    );
    return loaded;
  }

  /**
   * Pre-warms all remaining manifest poses (all 257 WLASL signs) in non-blocking
   * `requestIdleCallback` batches so live signing hits the synchronous cached fast-path.
   */
  prewarmAllPosesInIdle({ batchSize = 12, scheduleIdle } = {}) {
    const allGlosses = Object.keys(this.manifest?.poses || {});
    if (allGlosses.length === 0) return Promise.resolve(0);

    this._idlePrewarmCancelled = false;
    const schedule =
      scheduleIdle ||
      (typeof requestIdleCallback === "function"
        ? (cb) => requestIdleCallback(cb, { timeout: 600 })
        : (cb) => setTimeout(() => cb({ didTimeout: false, timeRemaining: () => 12 }), 24));

    let cursor = 0;
    let loadedCount = 0;

    return new Promise((resolve) => {
      const step = async () => {
        if (this._idlePrewarmCancelled) {
          resolve(loadedCount);
          return;
        }

        const batch = [];
        while (cursor < allGlosses.length && batch.length < batchSize) {
          const gloss = allGlosses[cursor++];
          const src = this.manifest.poses[gloss];
          if (src && !this.poseCache.has(src) && !this.inFlightPoses.has(src)) {
            batch.push(gloss);
          }
        }

        if (batch.length > 0) {
          loadedCount += await this.prefetchPoses(batch);
        }

        if (cursor < allGlosses.length && !this._idlePrewarmCancelled) {
          this._idlePrewarmHandle = schedule(step);
        } else {
          this._idlePrewarmHandle = null;
          resolve(loadedCount);
        }
      };

      this._idlePrewarmHandle = schedule(step);
    });
  }

  cancelIdlePrewarm() {
    this._idlePrewarmCancelled = true;
    if (this._idlePrewarmHandle !== null) {
      if (typeof cancelIdleCallback === "function") {
        cancelIdleCallback(this._idlePrewarmHandle);
      } else {
        clearTimeout(this._idlePrewarmHandle);
      }
      this._idlePrewarmHandle = null;
    }
  }
}
