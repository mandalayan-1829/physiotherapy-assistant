/**
 * Self-host the MediaPipe Tasks Vision WASM runtime.
 *
 * Why this exists
 * ---------------
 * `src/utils/poseEstimator.ts` initialises the landmarker with
 * `FilesetResolver.forVisionTasks('/mediapipe/wasm')`. Those files are *not*
 * bundled by Vite: they are plain `.js`/`.wasm` assets that the runtime fetches
 * at request time. Without this step `public/mediapipe/wasm` stays empty, the
 * fetch 404s in a clean checkout, and real pose inference fails for every user.
 *
 * The files are copied out of the installed `@mediapipe/tasks-vision` package so
 * the runtime version is locked to `package-lock.json` - nothing is fetched from
 * a third-party origin and no "latest" URL is used.
 *
 * Run automatically via the `predev` and `prebuild` npm scripts, or directly:
 *
 *     node scripts/copy-mediapipe-assets.mjs
 */

import { cpSync, existsSync, mkdirSync, readdirSync, rmSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const projectRoot = resolve(dirname(fileURLToPath(import.meta.url)), '..');

/** Must stay in step with `WASM_BASE_PATH` in `src/utils/poseEstimator.ts`. */
const DESTINATION = join(projectRoot, 'public', 'mediapipe', 'wasm');

/** Resolved from the installed package, so the version tracks package-lock.json. */
const SOURCE = join(projectRoot, 'node_modules', '@mediapipe', 'tasks-vision', 'wasm');

/**
 * The runtime `FilesetResolver` may fall back to the non-SIMD build, so both
 * variants must be present.
 */
const REQUIRED_FILES = [
  'vision_wasm_internal.js',
  'vision_wasm_internal.wasm',
  'vision_wasm_nosimd_internal.js',
  'vision_wasm_nosimd_internal.wasm',
];

function fail(message) {
  console.error(`\n[mediapipe-assets] ${message}\n`);
  process.exit(1);
}

if (!existsSync(SOURCE)) {
  fail(
    `MediaPipe runtime not found at ${SOURCE}.\n` +
      'The @mediapipe/tasks-vision package is missing or was not installed.\n' +
      'Run `npm install` (without --ignore-scripts) and build again.',
  );
}

const available = new Set(readdirSync(SOURCE));
const missing = REQUIRED_FILES.filter((file) => !available.has(file));
if (missing.length > 0) {
  fail(
    `The installed @mediapipe/tasks-vision package is missing expected runtime ` +
      `file(s): ${missing.join(', ')}.\n` +
      'Refusing to continue, because the app would silently fail to track a pose ' +
      'at request time rather than at build time.',
  );
}

// Rebuild the directory from scratch so a stale wasm from an older package
// version can never be served alongside a newer .js loader.
rmSync(DESTINATION, { recursive: true, force: true });
mkdirSync(DESTINATION, { recursive: true });
cpSync(SOURCE, DESTINATION, { recursive: true });

const copied = readdirSync(DESTINATION).sort();
console.log(
  `[mediapipe-assets] Copied ${copied.length} runtime file(s) to ` +
    `public/mediapipe/wasm: ${copied.join(', ')}`,
);
