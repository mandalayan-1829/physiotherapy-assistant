/**
 * Real on-device pose estimation.
 *
 * Runs MediaPipe's `PoseLandmarker` (WASM/WebGL) directly in the browser against
 * the camera stream. Nothing is uploaded: the frames are read from the
 * `<video>` element, converted to 33 normalised landmarks, and discarded. Only
 * the aggregate session metrics (reps, form score, duration) are ever sent to
 * the backend.
 *
 * This module replaces the previous `generatePoseLandmarks()` implementation,
 * which fabricated landmarks with a trigonometric model and must never be
 * presented as a measurement.
 *
 * Failure handling is explicit. If the model or runtime cannot be loaded, an
 * error is surfaced to the caller and no metrics are produced - the code never
 * silently falls back to synthetic data.
 */

import type { PoseLandmarker } from '@mediapipe/tasks-vision';
import type { Landmark } from '../types';

type VisionModule = typeof import('@mediapipe/tasks-vision');

/** MediaPipe Pose returns the 33-point BlazePose topology. */
export const POSE_LANDMARK_COUNT = 33;

/** A landmark must reach this visibility score to count towards a tracked frame. */
export const VISIBILITY_THRESHOLD = 0.5;

/**
 * Minimum share of landmarks that must be visible before a frame is treated as a
 * usable measurement. Frames below this are reported as "no pose" rather than
 * being fed to the biomechanics engine, so a half-tracked body can never produce
 * a rep or a form score.
 */
export const MIN_VISIBLE_RATIO = 0.5;

/**
 * Where the MediaPipe WASM runtime is served from. The assets are copied out of
 * `node_modules/@mediapipe/tasks-vision/wasm` into `public/mediapipe/wasm` by
 * `scripts/copy-mediapipe-assets.mjs` so they are self-hosted.
 */
const WASM_BASE_PATH: string = import.meta.env.VITE_POSE_WASM_PATH ?? '/mediapipe/wasm';

/**
 * The pose model itself. Defaults to Google's public model CDN; set
 * `VITE_POSE_MODEL_URL` to a self-hosted copy to remove the last external request
 * the application makes.
 */
const DEFAULT_MODEL_URL =
  'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task';

export const POSE_MODEL_URL: string = import.meta.env.VITE_POSE_MODEL_URL ?? DEFAULT_MODEL_URL;

export type PoseFailureReason = 'runtime' | 'model' | 'not_initialised';

export class PoseInferenceError extends Error {
  readonly reason: PoseFailureReason;

  constructor(reason: PoseFailureReason, message: string, cause?: unknown) {
    super(message, cause === undefined ? undefined : { cause });
    this.name = 'PoseInferenceError';
    this.reason = reason;
  }
}

/** One analysed camera frame. */
export interface PoseFrame {
  landmarks: Landmark[];
  /** Landmarks whose visibility score cleared {@link VISIBILITY_THRESHOLD}. */
  visibleLandmarks: number;
  visibleRatio: number;
  /** True when the frame is complete enough to be used as a measurement. */
  usable: boolean;
}

let cachedVisionModule: VisionModule | null = null;

async function loadVisionRuntime(): Promise<VisionModule> {
  if (cachedVisionModule) return cachedVisionModule;
  try {
    // Dynamic import keeps the multi-megabyte runtime out of the initial bundle.
    cachedVisionModule = await import('@mediapipe/tasks-vision');
    return cachedVisionModule;
  } catch (error) {
    throw new PoseInferenceError(
      'runtime',
      'The pose-estimation runtime could not be loaded.',
      error,
    );
  }
}

export interface PoseEstimatorOptions {
  /** Overrides the WASM location (tests / custom hosting). */
  wasmBasePath?: string;
  /** Overrides the model location (tests / self-hosted model). */
  modelUrl?: string;
}

export class PoseEstimator {
  private landmarker: PoseLandmarker | null = null;
  private readonly wasmBasePath: string;
  private readonly modelUrl: string;
  private lastTimestampMs = -1;
  private delegate: 'GPU' | 'CPU' | null = null;

  constructor(options: PoseEstimatorOptions = {}) {
    this.wasmBasePath = options.wasmBasePath ?? WASM_BASE_PATH;
    this.modelUrl = options.modelUrl ?? POSE_MODEL_URL;
  }

  get isReady(): boolean {
    return this.landmarker !== null;
  }

  get activeDelegate(): 'GPU' | 'CPU' | null {
    return this.delegate;
  }

  /** Load the runtime and the model. Safe to call repeatedly. */
  async initialise(): Promise<void> {
    if (this.landmarker) return;

    const vision = await loadVisionRuntime();

    let fileset;
    try {
      fileset = await vision.FilesetResolver.forVisionTasks(this.wasmBasePath);
    } catch (error) {
      throw new PoseInferenceError(
        'runtime',
        'The pose-estimation runtime could not be initialised.',
        error,
      );
    }

    const buildOptions = (delegate: 'GPU' | 'CPU') => ({
      baseOptions: {
        modelAssetPath: this.modelUrl,
        delegate,
      },
      runningMode: 'VIDEO' as const,
      numPoses: 1,
      minPoseDetectionConfidence: 0.5,
      minPosePresenceConfidence: 0.5,
      minTrackingConfidence: 0.5,
    });

    // Prefer the GPU delegate, but fall back to CPU on devices where WebGL is
    // unavailable (many mobile browsers and locked-down environments).
    try {
      this.landmarker = await vision.PoseLandmarker.createFromOptions(fileset, buildOptions('GPU'));
      this.delegate = 'GPU';
      return;
    } catch {
      try {
        this.landmarker = await vision.PoseLandmarker.createFromOptions(
          fileset,
          buildOptions('CPU'),
        );
        this.delegate = 'CPU';
      } catch (error) {
        throw new PoseInferenceError(
          'model',
          'The pose model could not be loaded. Check the network connection and try again.',
          error,
        );
      }
    }
  }

  /**
   * Analyse a single video frame.
   *
   * Returns `null` when no body is present, and a frame with `usable: false` when
   * only part of the body is visible. Neither case is allowed to produce a rep.
   */
  detect(video: HTMLVideoElement, timestampMs: number): PoseFrame | null {
    const landmarker = this.landmarker;
    if (!landmarker) {
      throw new PoseInferenceError('not_initialised', 'Pose estimation has not been initialised.');
    }
    // `detectForVideo` requires a monotonically increasing timestamp and a frame
    // with actual pixel data.
    if (video.readyState < 2 || video.videoWidth === 0 || video.videoHeight === 0) {
      return null;
    }

    const timestamp = Math.max(Math.floor(timestampMs), this.lastTimestampMs + 1);
    this.lastTimestampMs = timestamp;

    const result = landmarker.detectForVideo(video, timestamp);
    const detected = result.landmarks?.[0];
    if (!detected || detected.length === 0) return null;

    const landmarks: Landmark[] = detected.map((point) => ({
      x: point.x,
      y: point.y,
      z: point.z,
      visibility: point.visibility,
    }));

    const visibleLandmarks = landmarks.filter(
      (point) => (point.visibility ?? 1) >= VISIBILITY_THRESHOLD,
    ).length;
    const visibleRatio = landmarks.length > 0 ? visibleLandmarks / landmarks.length : 0;

    return {
      landmarks,
      visibleLandmarks,
      visibleRatio,
      usable: visibleRatio >= MIN_VISIBLE_RATIO,
    };
  }

  close(): void {
    try {
      this.landmarker?.close();
    } catch {
      // Closing a partially-initialised landmarker must never mask the real error.
    }
    this.landmarker = null;
    this.delegate = null;
    this.lastTimestampMs = -1;
  }
}

export function createPoseEstimator(options?: PoseEstimatorOptions): PoseEstimator {
  return new PoseEstimator(options);
}
