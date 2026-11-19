#!/usr/bin/env node
/**
 * Scans public/assets/letters/ and public/assets/poses/ to generate public/assets/manifest.json.
 * Run via: npm run build:manifest
 */

import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(__dirname, "..");
const ASSETS_DIR = path.join(ROOT, "public", "assets");
const LETTERS_DIR = path.join(ASSETS_DIR, "letters");
const POSES_DIR = path.join(ASSETS_DIR, "poses");
const ALIASES_PATH = path.join(ASSETS_DIR, "aliases.json");
const MANIFEST_PATH = path.join(ASSETS_DIR, "manifest.json");

function ensureDir(dir) {
  if (!fs.existsSync(dir)) {
    fs.mkdirSync(dir, { recursive: true });
  }
}

function loadAliases() {
  if (!fs.existsSync(ALIASES_PATH)) {
    return {};
  }
  try {
    const raw = JSON.parse(fs.readFileSync(ALIASES_PATH, "utf-8"));
    const normalized = {};
    for (const [alias, canonical] of Object.entries(raw)) {
      if (typeof alias === "string" && typeof canonical === "string" && !alias.startsWith("_")) {
        normalized[alias.trim().toUpperCase()] = canonical.trim().toUpperCase();
      }
    }
    return normalized;
  } catch (err) {
    console.warn(`[manifest] Could not parse aliases.json: ${err.message}`);
    return {};
  }
}

function buildManifest() {
  ensureDir(LETTERS_DIR);
  ensureDir(POSES_DIR);

  const letters = {};
  for (const file of fs.readdirSync(LETTERS_DIR).sort()) {
    const ext = path.extname(file).toLowerCase();
    const base = path.basename(file, ext).toUpperCase();
    if (/^[A-Z]$/.test(base) && (ext === ".svg" || ext === ".png")) {
      letters[base] = `/assets/letters/${file}`;
    }
  }

  const poses = {};
  for (const file of fs.readdirSync(POSES_DIR).sort()) {
    const ext = path.extname(file).toLowerCase();
    const base = path.basename(file, ext).toUpperCase().replace(/[\s-]+/g, "_");
    if (!base || base.startsWith(".") || ext !== ".json") continue;
    poses[base] = `/assets/poses/${file}`;
  }

  const aliases = loadAliases();

  const manifest = {
    version: 2,
    generatedAt: new Date().toISOString(),
    attribution: {
      poses: "Google Isolated Sign Language Recognition (DPAN & Georgia Tech, CC-BY 4.0)",
      letters: "Wikimedia Commons ASL Manual Alphabet (Public Domain)",
    },
    letterCount: Object.keys(letters).length,
    poseCount: Object.keys(poses).length,
    letters,
    poses,
    aliases,
  };

  fs.writeFileSync(MANIFEST_PATH, JSON.stringify(manifest, null, 2) + "\n", "utf-8");
  console.log(
    `[manifest] Built public/assets/manifest.json (${manifest.letterCount} letters, ${manifest.poseCount} poses, ${Object.keys(aliases).length} aliases)`
  );
}

buildManifest();
