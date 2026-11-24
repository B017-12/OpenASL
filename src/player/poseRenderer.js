/**
 * Live 2D Articulated Signer Renderer for MediaPipe Holistic Pose JSONs.
 * Draws real Deaf-signer landmark coordinates directly onto HTML5 <canvas>
 * inside a smoked-obsidian architectural viewfinder (zero pre-rendered videos,
 * zero rounded pills, high-contrast studio telemetry palette).
 */

// Standard MediaPipe 21-point hand bone connections
const HAND_CONNECTIONS = [
  // Thumb
  [0, 1], [1, 2], [2, 3], [3, 4],
  // Index
  [0, 5], [5, 6], [6, 7], [7, 8],
  // Middle
  [0, 9], [9, 10], [10, 11], [11, 12],
  // Ring
  [0, 13], [13, 14], [14, 15], [15, 16],
  // Pinky
  [0, 17], [17, 18], [18, 19], [19, 20],
  // Palm knuckle arch
  [5, 9], [9, 13], [13, 17],
];

const PALM_POLYGON = [0, 1, 5, 9, 13, 17];

function lerp(a, b, t) {
  return a + (b - a) * t;
}

function easeInOutCubic(t) {
  return t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2;
}

function interpolatePointArray(arrA, arrB, t) {
  if (!arrA && !arrB) return null;
  if (!arrA) return arrB;
  if (!arrB) return arrA;
  const len = Math.max(arrA.length, arrB.length);
  const out = new Array(len);
  for (let i = 0; i < len; i++) {
    const ptA = arrA[i];
    const ptB = arrB[i];
    if (ptA && ptB) {
      out[i] = [lerp(ptA[0], ptB[0], t), lerp(ptA[1], ptB[1], t)];
    } else {
      out[i] = t < 0.5 ? ptA || ptB || null : ptB || ptA || null;
    }
  }
  return out;
}

/**
 * Smoothly blends two MediaPipe landmark frames for natural co-articulation between signs.
 */
export function interpolatePoseFrames(frameA, frameB, rawT) {
  const t = easeInOutCubic(Math.max(0, Math.min(1, rawT)));
  if (!frameA) return frameB;
  if (!frameB) return frameA;
  return {
    p: interpolatePointArray(frameA.p, frameB.p, t),
    lh: interpolatePointArray(frameA.lh, frameB.lh, t),
    rh: interpolatePointArray(frameA.rh, frameB.rh, t),
    f: interpolatePointArray(frameA.f, frameB.f, t),
  };
}

function toCanvas(pt, rect) {
  if (!pt) return null;
  return [rect.x + pt[0] * rect.size, rect.y + pt[1] * rect.size];
}

function drawLimb(ctx, ptA, ptB, width, color) {
  if (!ptA || !ptB) return;
  ctx.strokeStyle = color;
  ctx.lineWidth = width;
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  ctx.beginPath();
  ctx.moveTo(ptA[0], ptA[1]);
  ctx.lineTo(ptB[0], ptB[1]);
  ctx.stroke();
}

function drawPolyline(ctx, pts, rect, closed = false) {
  const mapped = pts.map((p) => toCanvas(p, rect)).filter(Boolean);
  if (mapped.length < 2) return;
  ctx.beginPath();
  ctx.moveTo(mapped[0][0], mapped[0][1]);
  for (let i = 1; i < mapped.length; i++) {
    ctx.lineTo(mapped[i][0], mapped[i][1]);
  }
  if (closed) ctx.closePath();
  ctx.stroke();
}

function drawHand(ctx, handPts, rect, isDominant) {
  if (!handPts) return;
  const mapped = handPts.map((pt) => toCanvas(pt, rect));

  // Filled palm silhouette (dominant = studio phosphor-teal, non-dominant = WebAI signal amber)
  const palmPts = PALM_POLYGON.map((idx) => mapped[idx]).filter(Boolean);
  if (palmPts.length >= 4) {
    ctx.fillStyle = isDominant ? "rgba(42, 195, 162, 0.22)" : "rgba(240, 124, 24, 0.20)";
    ctx.beginPath();
    ctx.moveTo(palmPts[0][0], palmPts[0][1]);
    for (let i = 1; i < palmPts.length; i++) {
      ctx.lineTo(palmPts[i][0], palmPts[i][1]);
    }
    ctx.closePath();
    ctx.fill();
  }

  // Finger bones calibrated for dark smoked-obsidian legibility (zero halation)
  const boneColor = isDominant ? "#36CFAF" : "#F08524";
  const jointColor = isDominant ? "#89F0DA" : "#FFC184";

  ctx.strokeStyle = boneColor;
  ctx.lineCap = "round";
  ctx.lineJoin = "round";

  for (const [a, b] of HAND_CONNECTIONS) {
    const pa = mapped[a];
    const pb = mapped[b];
    if (!pa || !pb) continue;
    const isTip = b % 4 === 0 && b > 0;
    ctx.lineWidth = isTip ? 4.0 : 5.2;
    ctx.beginPath();
    ctx.moveTo(pa[0], pa[1]);
    ctx.lineTo(pb[0], pb[1]);
    ctx.stroke();
  }

  // Articulated joint markers
  for (let i = 0; i < mapped.length; i++) {
    const pt = mapped[i];
    if (!pt) continue;
    const isTip = i % 4 === 0 && i > 0;
    ctx.fillStyle = isTip ? "#EAECEF" : jointColor;
    ctx.beginPath();
    ctx.arc(pt[0], pt[1], isTip ? 3.1 : 2.5, 0, Math.PI * 2);
    ctx.fill();
  }
}

/**
 * Renders one normalized MediaPipe pose frame inside `rect = { x, y, size }`.
 */
export function renderPoseFrame(ctx, frame, rect) {
  if (!frame) return;

  ctx.save();
  ctx.beginPath();
  ctx.rect(rect.x + 2, rect.y + 2, rect.size - 4, rect.size - 4);
  ctx.clip();

  const p = frame.p || [];
  // POSE_INDICES order: [0: nose, 1: L_shoulder, 2: R_shoulder, 3: L_elbow, 4: R_elbow, 5: L_wrist, 6: R_wrist]
  const nose = toCanvas(p[0], rect);
  const lShoulder = toCanvas(p[1], rect);
  const rShoulder = toCanvas(p[2], rect);
  const lElbow = toCanvas(p[3], rect);
  const rElbow = toCanvas(p[4], rect);
  const lWrist = toCanvas(p[5], rect);
  const rWrist = toCanvas(p[6], rect);

  // 1. Stylized torso & head silhouette anchor on smoked-obsidian
  if (lShoulder && rShoulder) {
    const midX = (lShoulder[0] + rShoulder[0]) * 0.5;
    const midY = (lShoulder[1] + rShoulder[1]) * 0.5;
    const shoulderW = Math.hypot(lShoulder[0] - rShoulder[0], lShoulder[1] - rShoulder[1]);

    // Subtle architectural torso silhouette
    ctx.fillStyle = "rgba(234, 236, 239, 0.055)";
    ctx.beginPath();
    ctx.moveTo(lShoulder[0], lShoulder[1]);
    ctx.lineTo(rShoulder[0], rShoulder[1]);
    ctx.lineTo(rShoulder[0] - shoulderW * 0.06, rect.y + rect.size);
    ctx.lineTo(lShoulder[0] + shoulderW * 0.06, rect.y + rect.size);
    ctx.closePath();
    ctx.fill();

    // Shoulder line
    drawLimb(ctx, lShoulder, rShoulder, 9.5, "rgba(148, 158, 174, 0.34)");

    // Head outline anchored to nose / shoulder midpoint
    const headX = nose ? nose[0] : midX;
    const headY = nose ? nose[1] - shoulderW * 0.04 : midY - shoulderW * 0.52;
    const headRx = Math.max(26, shoulderW * 0.24);
    const headRy = headRx * 1.24;

    ctx.fillStyle = "rgba(234, 236, 239, 0.045)";
    ctx.strokeStyle = "rgba(234, 236, 239, 0.30)";
    ctx.lineWidth = 2.2;
    ctx.beginPath();
    ctx.ellipse(headX, headY, headRx, headRy, 0, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();
  }

  // 2. Facial non-manual grammar cues (eyebrows, eyes, mouth)
  if (frame.f && frame.f.length >= 22) {
    ctx.strokeStyle = "rgba(234, 236, 239, 0.72)";
    ctx.lineWidth = 2.3;
    ctx.lineCap = "round";
    // Eyebrows (critical for WH vs Yes/No questions in ASL)
    drawPolyline(ctx, frame.f.slice(0, 5), rect, false);
    drawPolyline(ctx, frame.f.slice(5, 10), rect, false);
    // Eyes
    ctx.lineWidth = 1.7;
    drawPolyline(ctx, frame.f.slice(10, 14), rect, true);
    drawPolyline(ctx, frame.f.slice(14, 18), rect, true);
    // Mouth
    ctx.lineWidth = 2.1;
    drawPolyline(ctx, frame.f.slice(18, 22), rect, true);
  }

  // 3. Arms (connect elbows to hand wrist if hand is visible, else pose wrist)
  const lhWrist = frame.lh?.[0] ? toCanvas(frame.lh[0], rect) : lWrist;
  const rhWrist = frame.rh?.[0] ? toCanvas(frame.rh[0], rect) : rWrist;

  drawLimb(ctx, lShoulder, lElbow, 10, "rgba(148, 158, 174, 0.36)");
  drawLimb(ctx, lElbow, lhWrist, 8.5, "rgba(200, 208, 220, 0.44)");

  drawLimb(ctx, rShoulder, rElbow, 10, "rgba(148, 158, 174, 0.36)");
  drawLimb(ctx, rElbow, rhWrist, 8.5, "rgba(200, 208, 220, 0.44)");

  // 4. Articulated Hands (foreground layer)
  drawHand(ctx, frame.lh, rect, false);
  drawHand(ctx, frame.rh, rect, true);

  ctx.restore();
}
