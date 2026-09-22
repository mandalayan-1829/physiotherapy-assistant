/**
 * Download the MediaPipe pose model so it can be self-hosted.
 *
 * The app works without this: `src/utils/poseEstimator.ts` defaults to Google's
 * version-pinned model CDN (see `DEFAULT_MODEL_URL`). Running this script removes
 * that last outbound request, which is what you want if the app must run offline,
 * on a locked-down network, or under a policy that forbids third-party fetches
 * from the browser.
 *
 * The upstream path is pinned by version (`.../float16/1/...`). It is deliberately
 * *not* a "latest" URL: a model that silently changes under a research pipeline
 * would invalidate every measurement recorded before the change.
 *
 *     node scripts/fetch-pose-model.mjs
 *     # then set in .env:
 *     #   VITE_POSE_MODEL_URL=/mediapipe/models/pose_landmarker_lite.task
 *
 * The result lands in `public/mediapipe/models/`, which is git-ignored: it is a
 * large binary that is reproducible from the pinned URL below.
 */

import { mkdirSync, statSync, writeFileSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const projectRoot = resolve(dirname(fileURLToPath(import.meta.url)), '..');

/** Must match `DEFAULT_MODEL_URL` in src/utils/poseEstimator.ts. */
const MODEL_URL =
  'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task';

const DESTINATION = join(
  projectRoot,
  'public',
  'mediapipe',
  'models',
  'pose_landmarker_lite.task',
);

/** A truncated download is worse than no download, so sanity-check the size. */
const MIN_EXPECTED_BYTES = 1_000_000;

console.log(`[pose-model] Downloading ${MODEL_URL}`);

let response;
try {
  response = await fetch(MODEL_URL);
} catch (error) {
  console.error(
    `[pose-model] Download failed: ${error?.message ?? error}\n` +
      'The app still works - it will load the model from the public CDN instead.',
  );
  process.exit(1);
}

if (!response.ok) {
  console.error(
    `[pose-model] Download failed with HTTP ${response.status} ${response.statusText}.\n` +
      'The app still works - it will load the model from the public CDN instead.',
  );
  process.exit(1);
}

const bytes = Buffer.from(await response.arrayBuffer());

if (bytes.byteLength < MIN_EXPECTED_BYTES) {
  console.error(
    `[pose-model] Refusing to write a ${bytes.byteLength}-byte response; the ` +
      'download looks truncated or the URL returned an error page.',
  );
  process.exit(1);
}

mkdirSync(dirname(DESTINATION), { recursive: true });
writeFileSync(DESTINATION, bytes);

console.log(
  `[pose-model] Wrote ${(statSync(DESTINATION).size / 1_048_576).toFixed(1)} MB to ` +
    `${DESTINATION}\n` +
    '[pose-model] Now set VITE_POSE_MODEL_URL=/mediapipe/models/pose_landmarker_lite.task',
);
