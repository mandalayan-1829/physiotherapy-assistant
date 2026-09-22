import { Landmark } from '../types';

export interface Point2D {
  x: number;
  y: number;
}

/**
 * Calculates the angle at point B, formed by points A-B-C.
 *
 * Coordinate system
 * ----------------
 * MediaPipe returns 33 landmarks normalised to the frame: `x` and `y` are
 * fractions of the image width and height in [0, 1], and `y` increases
 * *downwards*. `z` is roughly the same scale as `x`.
 *
 * Because the coordinates are normalised, absolute distances depend on how much
 * of the frame the subject fills. The angle here is therefore scale-invariant,
 * which is why it - rather than a raw joint distance - is the primary feature for
 * the exercise state machines. Anything that *does* depend on a distance is
 * normalised first (see the torso/limb ratios in `exerciseEngine.ts`).
 *
 * Math
 * ----
 * With BA = A - B and BC = C - B (vectors pointing away from the vertex B):
 *
 *     cos(theta) = dot(BA, BC) / (|BA| * |BC|)
 *     theta      = arccos(clamp(cos(theta), -1, 1))
 *
 * The clamp guards against floating-point drift pushing the cosine marginally
 * outside [-1, 1], which would otherwise make `arccos` return NaN and poison
 * every downstream feature for that frame.
 *
 * Returns the angle in degrees, rounded to one decimal place.
 */
export function calculateAngle(a: Point2D | [number, number], b: Point2D | [number, number], c: Point2D | [number, number]): number {
  const ax = Array.isArray(a) ? a[0] : a.x;
  const ay = Array.isArray(a) ? a[1] : a.y;
  const bx = Array.isArray(b) ? b[0] : b.x;
  const by = Array.isArray(b) ? b[1] : b.y;
  const cx = Array.isArray(c) ? c[0] : c.x;
  const cy = Array.isArray(c) ? c[1] : c.y;

  const bax = ax - bx;
  const bay = ay - by;
  const bcx = cx - bx;
  const bcy = cy - by;

  const dot = bax * bcx + bay * bcy;
  const magBA = Math.sqrt(bax * bax + bay * bay);
  const magBC = Math.sqrt(bcx * bcx + bcy * bcy);

  if (magBA * magBC === 0) return 0;

  let cosTheta = dot / (magBA * magBC + 1e-6);
  cosTheta = Math.max(-1.0, Math.min(1.0, cosTheta));

  const angleRad = Math.acos(cosTheta);
  const angleDeg = (angleRad * 180) / Math.PI;

  return Math.round(angleDeg * 10) / 10;
}

// MediaPipe standard landmark indices
export const POSE_LANDMARKS = {
  NOSE: 0,
  LEFT_EYE_INNER: 1,
  LEFT_EYE: 2,
  LEFT_EYE_OUTER: 3,
  RIGHT_EYE_INNER: 4,
  RIGHT_EYE: 5,
  RIGHT_EYE_OUTER: 6,
  LEFT_EAR: 7,
  RIGHT_EAR: 8,
  MOUTH_LEFT: 9,
  MOUTH_RIGHT: 10,
  LEFT_SHOULDER: 11,
  RIGHT_SHOULDER: 12,
  LEFT_ELBOW: 13,
  RIGHT_ELBOW: 14,
  LEFT_WRIST: 15,
  RIGHT_WRIST: 16,
  LEFT_PINKY: 17,
  RIGHT_PINKY: 18,
  LEFT_INDEX: 19,
  RIGHT_INDEX: 20,
  LEFT_THUMB: 21,
  RIGHT_THUMB: 22,
  LEFT_HIP: 23,
  RIGHT_HIP: 24,
  LEFT_KNEE: 25,
  RIGHT_KNEE: 26,
  LEFT_ANKLE: 27,
  RIGHT_ANKLE: 28,
  LEFT_HEEL: 29,
  RIGHT_HEEL: 30,
  LEFT_FOOT_INDEX: 31,
  RIGHT_FOOT_INDEX: 32,
} as const;

export function getLandmarkCoords(landmarks: Landmark[], index: number): [number, number] {
  if (!landmarks || !landmarks[index]) {
    return [0, 0];
  }
  return [landmarks[index].x, landmarks[index].y];
}
