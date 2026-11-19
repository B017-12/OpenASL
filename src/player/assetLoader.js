/**
 * Loads public/assets/manifest.json, preloads A-Z fingerspelling SVGs,
 * and caches MediaPipe Holistic pose JSON sequences (from Google asl-signs or local drop-in).
 */

function loadImage(src) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = () => reject(new Error(`Failed to load image: ${src}`));
    img.src = src;
  });
}

export class ASLAssetStore {
  constructor() {
    this.manifest = { letters: {}, poses: {}, aliases: {} };
    this.letterImages = new Map();
    this.poseCache = new Map();
    this.localPoseOverrides = new Map(); // Local .json pose sequences dropped in for preview
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
    return this.letterImages.get(letter.toUpperCase()) || null;
  }

  resolvePoseEntry(gloss) {
    const key = gloss.toUpperCase().trim();
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

  async loadPoseData(entry) {
    if (!entry) return null;
    if (entry.data) return entry.data;
    if (this.poseCache.has(entry.src)) {
      return this.poseCache.get(entry.src);
    }

    const resp = await fetch(entry.src);
    if (!resp.ok) {
      throw new Error(`Failed to load pose JSON: ${entry.src}`);
    }
    const data = await resp.json();
    if (!Array.isArray(data.frames) || data.frames.length === 0) {
      throw new Error(`Pose JSON has no frames: ${entry.src}`);
    }
    this.poseCache.set(entry.src, data);
    return data;
  }

  /**
   * Registers a locally dropped pose JSON file (exported by extract_asl_signs.py) for preview.
   */
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

    this.localPoseOverrides.set(key, {
      ...parsed,
      gloss: key,
      fileName: file.name,
    });
    return { gloss: key, frameCount: parsed.frames.length, fileName: file.name };
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
}
