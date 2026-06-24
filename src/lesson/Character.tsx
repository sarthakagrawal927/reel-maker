import { interpolate, useCurrentFrame } from "remotion";

export type Pose = "idle" | "wave" | "hold" | "cheer" | "point";

/**
 * A friendly parametric "code buddy" rendered as SVG. Personifies a closure:
 * it can hold a `value` box (the captured variable) in its hands.
 */
export const Character: React.FC<{
  color?: string;
  glow?: string;
  pose?: Pose;
  eyeDir?: [number, number]; // -1..1
  holdValue?: string | null;
  holdEnter?: number; // 0..1 — animates the captured box dropping into hands
  size?: number;
  seed?: number; // varies idle bob phase between instances
}> = ({ color = "#38bdf8", glow = "#38bdf8", pose = "idle", eyeDir = [0, 0], holdValue = null, holdEnter = 1, size = 360, seed = 0 }) => {
  const frame = useCurrentFrame();
  const bob = Math.sin((frame + seed * 20) / 14) * 6;
  const blink = (frame + seed * 13) % 110 < 5 ? 0.12 : 1; // quick blink
  const waveAngle = pose === "wave" ? Math.sin(frame / 4) * 22 - 6 : 0;
  const cheer = pose === "cheer" ? Math.abs(Math.sin(frame / 6)) * 16 : 0;

  // arm rotations per pose (left, right) in degrees
  const armL = pose === "wave" ? waveAngle : pose === "cheer" ? -50 - cheer : pose === "hold" ? 38 : 14;
  const armR = pose === "hold" ? -38 : pose === "cheer" ? 50 + cheer : pose === "point" ? -64 : -14;

  const [ex, ey] = eyeDir;

  return (
    <div style={{ position: "relative", width: size, height: size, transform: `translateY(${bob}px)` }}>
      <svg viewBox="0 0 200 200" width={size} height={size} style={{ overflow: "visible" }}>
        <defs>
          <linearGradient id={`body-${seed}`} x1="0" y1="0" x2="0" y2="1">
            <stop offset="0" stopColor={color} />
            <stop offset="1" stopColor={color} stopOpacity={0.78} />
          </linearGradient>
          <filter id={`soft-${seed}`} x="-50%" y="-50%" width="200%" height="200%">
            <feDropShadow dx="0" dy="10" stdDeviation="12" floodColor={glow} floodOpacity="0.45" />
          </filter>
        </defs>

        {/* shadow */}
        <ellipse cx="100" cy="188" rx="52" ry="9" fill="rgba(0,0,0,0.35)" />

        {/* arms (behind body) */}
        <g stroke={color} strokeWidth="11" strokeLinecap="round" opacity="0.95">
          <g transform={`rotate(${armL} 50 112)`}>
            <line x1="50" y1="112" x2="24" y2="150" />
          </g>
          <g transform={`rotate(${armR} 150 112)`}>
            <line x1="150" y1="112" x2="176" y2="150" />
          </g>
        </g>

        {/* body */}
        <g filter={`url(#soft-${seed})`}>
          <rect x="46" y="70" width="108" height="100" rx="30" fill={`url(#body-${seed})`} />
        </g>
        {/* belly screen */}
        <rect x="62" y="92" width="76" height="58" rx="14" fill="#0a0e1a" opacity="0.85" />
        {/* face on the screen */}
        <g>
          {/* eyes */}
          <g transform={`translate(${ex * 5}, ${ey * 4})`}>
            <ellipse cx="86" cy="116" rx="9" ry={9 * blink} fill="#e7ecf5" />
            <ellipse cx="114" cy="116" rx="9" ry={9 * blink} fill="#e7ecf5" />
            <circle cx={86 + ex * 3} cy={116 + ey * 3} r="3.4" fill="#0a0e1a" />
            <circle cx={114 + ex * 3} cy={116 + ey * 3} r="3.4" fill="#0a0e1a" />
          </g>
          {/* smile */}
          <path d="M88 134 Q100 144 112 134" stroke={glow} strokeWidth="4" fill="none" strokeLinecap="round" />
        </g>
        {/* antenna */}
        <line x1="100" y1="70" x2="100" y2="52" stroke={color} strokeWidth="6" strokeLinecap="round" />
        <circle cx="100" cy="48" r="8" fill={glow}>
        </circle>
      </svg>

      {/* held value box (the captured variable) */}
      {holdValue != null && (
        <div
          style={{
            position: "absolute",
            left: "50%",
            top: size * 0.74,
            transform: "translateX(-50%)",
            background: "#0a0e1a",
            border: `2px solid ${glow}`,
            borderRadius: 14,
            padding: "8px 18px",
            fontFamily: "monospace",
            fontWeight: 700,
            fontSize: size * 0.12,
            color: glow,
            boxShadow: `0 0 24px ${glow}77`,
            whiteSpace: "nowrap",
          }}
        >
          {holdValue}
        </div>
      )}
    </div>
  );
};

/** Floating code-symbol particles for a lively background. */
export const Particles: React.FC<{ accent: string }> = ({ accent }) => {
  const frame = useCurrentFrame();
  const glyphs = ["{ }", "()", "=>", ";", "[]", "++", "//", "const", "fn", "{}"];
  const items = Array.from({ length: 16 }, (_, i) => {
    const x = ((i * 137) % 100);
    const baseY = ((i * 53) % 100);
    const drift = ((frame * (0.18 + (i % 5) * 0.05)) % 120) - 10;
    const y = (baseY + drift) % 110 - 5;
    const op = 0.05 + ((i % 4) * 0.03);
    const rot = Math.sin((frame + i * 30) / 40) * 12;
    return { g: glyphs[i % glyphs.length], x, y, op, rot, s: 26 + (i % 4) * 12 };
  });
  return (
    <>
      {items.map((it, i) => (
        <div
          key={i}
          style={{
            position: "absolute",
            left: `${it.x}%`,
            top: `${it.y}%`,
            fontFamily: "monospace",
            fontWeight: 700,
            fontSize: it.s,
            color: accent,
            opacity: it.op,
            transform: `rotate(${it.rot}deg)`,
          }}
        >
          {it.g}
        </div>
      ))}
    </>
  );
};
