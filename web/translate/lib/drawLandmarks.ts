/** Canvas overlay for MediaPipe holistic landmarks (server-normalized coords, mirror for selfie UI). */

export interface LandmarkPoint {
  x: number;
  y: number;
  z?: number;
  visibility?: number;
}

export interface HolisticLandmarks {
  pose: LandmarkPoint[] | null;
  face: LandmarkPoint[] | null;
  left_hand: LandmarkPoint[] | null;
  right_hand: LandmarkPoint[] | null;
}

export interface HandsDetected {
  left: boolean;
  right: boolean;
}

/** MediaPipe Pose (33 landmarks) */
const POSE_CONNECTIONS: [number, number][] = [
  [0, 1], [1, 2], [2, 3], [3, 7], [0, 4], [4, 5], [5, 6], [6, 8], [9, 10],
  [11, 12], [11, 13], [13, 15], [15, 17], [15, 19], [15, 21], [17, 19],
  [12, 14], [14, 16], [16, 18], [16, 20], [16, 22], [18, 20],
  [11, 23], [12, 24], [23, 24], [23, 25], [24, 26], [25, 27], [26, 28],
  [27, 29], [28, 30], [29, 31], [30, 32], [27, 31], [28, 32],
];

/** MediaPipe Hand (21 landmarks) */
const HAND_CONNECTIONS: [number, number][] = [
  [0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8],
  [0, 9], [9, 10], [10, 11], [11, 12], [0, 13], [13, 14], [14, 15], [15, 16],
  [0, 17], [17, 18], [18, 19], [19, 20], [5, 9], [9, 13], [13, 17],
];

/** Face oval + lips (subset of MediaPipe face mesh contours) */
const FACE_OVAL: number[] = [
  10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378,
  400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54,
  103, 67, 109, 10,
];

const FACE_LIPS: [number, number][] = [
  [61, 146], [146, 91], [91, 181], [181, 84], [84, 17], [17, 314], [314, 405],
  [405, 321], [321, 375], [375, 291], [61, 185], [185, 40], [40, 39], [39, 37],
  [37, 0], [0, 267], [267, 269], [269, 270], [270, 409], [409, 291], [78, 95],
  [95, 88], [88, 178], [178, 87], [87, 14], [14, 317], [317, 402], [402, 318],
  [318, 324], [324, 308], [78, 191], [191, 80], [80, 81], [81, 82], [82, 13],
  [13, 312], [312, 311], [311, 310], [310, 415], [415, 308],
];

const LEFT_EYE: [number, number][] = [
  [33, 7], [7, 163], [163, 144], [144, 145], [145, 153], [153, 154], [154, 155],
  [155, 133], [133, 173], [173, 157], [157, 158], [158, 159], [159, 160], [160, 161],
  [161, 246], [246, 33],
];

const RIGHT_EYE: [number, number][] = [
  [263, 249], [249, 390], [390, 373], [373, 374], [374, 380], [380, 381], [381, 382],
  [382, 362], [362, 398], [398, 384], [384, 385], [385, 386], [386, 387], [387, 388],
  [388, 466], [466, 263],
];

function parsePoint(raw: number[]): LandmarkPoint {
  return {
    x: raw[0],
    y: raw[1],
    z: raw[2],
    visibility: raw.length > 3 ? raw[3] : undefined,
  };
}

export type RawHolisticLandmarks = {
  pose?: number[][] | null;
  face?: number[][] | null;
  left_hand?: number[][] | null;
  right_hand?: number[][] | null;
};

export function parseHolisticLandmarks(
  raw: RawHolisticLandmarks | null | undefined,
): HolisticLandmarks | null {
  if (!raw) return null;
  const map = (key: keyof HolisticLandmarks) => {
    const arr = raw[key];
    if (!arr?.length) return null;
    return arr.map((row) => parsePoint(row));
  };
  return {
    pose: map("pose"),
    face: map("face"),
    left_hand: map("left_hand"),
    right_hand: map("right_hand"),
  };
}

function toCanvas(
  p: LandmarkPoint,
  width: number,
  height: number,
  mirror: boolean,
): [number, number] {
  const x = mirror ? (1 - p.x) * width : p.x * width;
  return [x, p.y * height];
}

function drawConnections(
  ctx: CanvasRenderingContext2D,
  points: LandmarkPoint[],
  connections: [number, number][],
  width: number,
  height: number,
  mirror: boolean,
  color: string,
  lineWidth: number,
) {
  ctx.strokeStyle = color;
  ctx.lineWidth = lineWidth;
  ctx.beginPath();
  for (const [a, b] of connections) {
    if (a >= points.length || b >= points.length) continue;
    const [x0, y0] = toCanvas(points[a], width, height, mirror);
    const [x1, y1] = toCanvas(points[b], width, height, mirror);
    ctx.moveTo(x0, y0);
    ctx.lineTo(x1, y1);
  }
  ctx.stroke();
}

function drawPath(
  ctx: CanvasRenderingContext2D,
  points: LandmarkPoint[],
  indices: number[],
  width: number,
  height: number,
  mirror: boolean,
  color: string,
  lineWidth: number,
) {
  if (indices.length < 2) return;
  ctx.strokeStyle = color;
  ctx.lineWidth = lineWidth;
  ctx.beginPath();
  for (let i = 0; i < indices.length; i++) {
    const idx = indices[i];
    if (idx >= points.length) continue;
    const [x, y] = toCanvas(points[idx], width, height, mirror);
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  }
  ctx.closePath();
  ctx.stroke();
}

function drawPoints(
  ctx: CanvasRenderingContext2D,
  points: LandmarkPoint[],
  width: number,
  height: number,
  mirror: boolean,
  color: string,
  radius: number,
) {
  ctx.fillStyle = color;
  for (const p of points) {
    const vis = p.visibility;
    if (vis !== undefined && vis < 0.5) continue;
    const [x, y] = toCanvas(p, width, height, mirror);
    ctx.beginPath();
    ctx.arc(x, y, radius, 0, Math.PI * 2);
    ctx.fill();
  }
}

export function hasAnyLandmarks(landmarks: HolisticLandmarks | null): boolean {
  if (!landmarks) return false;
  return Boolean(
    landmarks.pose?.length ||
      landmarks.face?.length ||
      landmarks.left_hand?.length ||
      landmarks.right_hand?.length,
  );
}

export function drawHolisticOverlay(
  ctx: CanvasRenderingContext2D,
  landmarks: HolisticLandmarks | null,
  width: number,
  height: number,
  options: { mirror?: boolean } = {},
): void {
  const mirror = options.mirror !== false;
  ctx.clearRect(0, 0, width, height);

  if (!landmarks) return;

  if (landmarks.pose?.length) {
    drawConnections(ctx, landmarks.pose, POSE_CONNECTIONS, width, height, mirror, "rgba(120, 140, 255, 0.95)", 3);
    drawPoints(ctx, landmarks.pose, width, height, mirror, "rgba(200, 210, 255, 1)", 3);
  }

  if (landmarks.face?.length) {
    drawPath(ctx, landmarks.face, FACE_OVAL, width, height, mirror, "rgba(200, 180, 255, 0.85)", 2);
    drawConnections(ctx, landmarks.face, FACE_LIPS, width, height, mirror, "rgba(220, 160, 255, 0.9)", 1.5);
    drawConnections(ctx, landmarks.face, LEFT_EYE, width, height, mirror, "rgba(200, 200, 255, 0.9)", 1.5);
    drawConnections(ctx, landmarks.face, RIGHT_EYE, width, height, mirror, "rgba(200, 200, 255, 0.9)", 1.5);
    drawPoints(ctx, landmarks.face, width, height, mirror, "rgba(230, 210, 255, 0.55)", 2);
  }

  if (landmarks.left_hand?.length) {
    drawConnections(ctx, landmarks.left_hand, HAND_CONNECTIONS, width, height, mirror, "rgba(80, 255, 120, 0.98)", 3);
    drawPoints(ctx, landmarks.left_hand, width, height, mirror, "rgba(120, 255, 160, 1)", 4);
  }

  if (landmarks.right_hand?.length) {
    drawConnections(ctx, landmarks.right_hand, HAND_CONNECTIONS, width, height, mirror, "rgba(80, 220, 255, 0.98)", 3);
    drawPoints(ctx, landmarks.right_hand, width, height, mirror, "rgba(120, 240, 255, 1)", 4);
  }
}
