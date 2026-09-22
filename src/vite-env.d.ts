/// <reference types="vite/client" />

interface ImportMetaEnv {
  /** Base URL of the FastAPI backend, e.g. http://localhost:8000 */
  readonly VITE_API_URL?: string;
  /**
   * Where the MediaPipe WASM runtime is served from. Defaults to
   * `/mediapipe/wasm`, which is populated by
   * `scripts/copy-mediapipe-assets.mjs`. Only change this if you host the
   * runtime somewhere else.
   */
  readonly VITE_POSE_WASM_PATH?: string;
  /**
   * Location of the pose model (`*.task`). Defaults to Google's public model
   * CDN; point this at a self-hosted copy to remove the last third-party
   * request the application makes.
   */
  readonly VITE_POSE_MODEL_URL?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
