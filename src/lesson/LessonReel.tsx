import { z } from "zod";
import {
  AbsoluteFill,
  Audio,
  Sequence,
  interpolate,
  spring,
  staticFile,
  useCurrentFrame,
  useVideoConfig,
} from "remotion";
import { loadFont as loadInter } from "@remotion/google-fonts/Inter";
import { loadFont as loadMono } from "@remotion/google-fonts/RobotoMono";

const { fontFamily: SANS } = loadInter("normal", {
  weights: ["500", "700", "800", "900"],
  subsets: ["latin"],
  ignoreTooManyRequestsWarning: true,
});
const { fontFamily: MONO } = loadMono("normal", {
  weights: ["400", "700"],
  subsets: ["latin"],
  ignoreTooManyRequestsWarning: true,
});

export const lessonSchema = z.object({ lesson: z.any().nullable() });

const C = {
  bg0: "#0a0e1a",
  bg1: "#111a2e",
  panel: "rgba(17,26,46,0.72)",
  border: "rgba(120,160,255,0.18)",
  accent: "#38bdf8",
  accent2: "#34d399",
  text: "#e7ecf5",
  dim: "#8a97b5",
  // syntax
  kw: "#c792ea",
  ret: "#c792ea",
  fn: "#82aaff",
  var: "#f78c6c",
  num: "#f78c6c",
  punct: "#89ddff",
  comment: "#5c6a86",
  plain: "#d6deeb",
};

const ms2f = (ms: number, fps: number) => Math.round((ms / 1000) * fps);

// ---------- shared chrome ----------

const Background: React.FC = () => {
  const frame = useCurrentFrame();
  const drift = Math.sin(frame / 80) * 14;
  return (
    <AbsoluteFill>
      <AbsoluteFill style={{ background: `radial-gradient(120% 90% at 50% 0%, ${C.bg1} 0%, ${C.bg0} 70%)` }} />
      <AbsoluteFill
        style={{
          backgroundImage:
            "linear-gradient(rgba(120,160,255,0.05) 1px, transparent 1px), linear-gradient(90deg, rgba(120,160,255,0.05) 1px, transparent 1px)",
          backgroundSize: "64px 64px",
          transform: `translateY(${drift}px)`,
          maskImage: "radial-gradient(80% 60% at 50% 40%, black 0%, transparent 90%)",
        }}
      />
      <AbsoluteFill
        style={{
          background: `radial-gradient(60% 30% at 50% -5%, ${C.accent}22 0%, transparent 70%)`,
        }}
      />
    </AbsoluteFill>
  );
};

const Brand: React.FC<{ handle: string; title: string }> = ({ handle, title }) => (
  <div
    style={{
      position: "absolute",
      top: 70,
      left: 0,
      right: 0,
      display: "flex",
      justifyContent: "center",
      alignItems: "center",
      gap: 16,
      fontFamily: SANS,
    }}
  >
    <div style={{ width: 14, height: 14, borderRadius: 4, background: C.accent2, boxShadow: `0 0 18px ${C.accent2}` }} />
    <span style={{ color: C.text, fontSize: 30, fontWeight: 700, letterSpacing: 0.5 }}>{title}</span>
    <span style={{ color: C.dim, fontSize: 28, fontWeight: 500 }}>· {handle}</span>
  </div>
);

const Progress: React.FC<{ totalMs: number }> = ({ totalMs }) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const pct = Math.min(1, (frame / fps) * 1000 / totalMs);
  return (
    <div style={{ position: "absolute", left: 80, right: 80, bottom: 70, height: 8, borderRadius: 99, background: "rgba(255,255,255,0.10)" }}>
      <div
        style={{
          width: `${pct * 100}%`,
          height: "100%",
          borderRadius: 99,
          background: `linear-gradient(90deg, ${C.accent}, ${C.accent2})`,
          boxShadow: `0 0 16px ${C.accent}88`,
        }}
      />
    </div>
  );
};

// ---------- captions (word-level, grouped into pages) ----------

type Cap = { text: string; startMs: number; endMs: number };

const buildPages = (caps: Cap[], maxWords = 4, maxMs = 1500) => {
  const pages: { words: Cap[]; startMs: number; endMs: number }[] = [];
  let cur: Cap[] = [];
  for (const w of caps) {
    if (cur.length && (cur.length >= maxWords || w.endMs - cur[0].startMs > maxMs)) {
      pages.push({ words: cur, startMs: cur[0].startMs, endMs: cur[cur.length - 1].endMs });
      cur = [];
    }
    cur.push(w);
  }
  if (cur.length) pages.push({ words: cur, startMs: cur[0].startMs, endMs: cur[cur.length - 1].endMs });
  return pages;
};

const Captions: React.FC<{ capGroups: Cap[][] }> = ({ capGroups }) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const nowMs = (frame / fps) * 1000;
  // Build pages per scene so a page never spans the pause between scenes.
  const pages = capGroups.flatMap((g) => buildPages(g));
  const page = pages.find((p) => nowMs >= p.startMs && nowMs <= p.endMs + 220) ?? null;
  if (!page) return null;
  const enter = interpolate(nowMs, [page.startMs, page.startMs + 130], [0, 1], { extrapolateLeft: "clamp", extrapolateRight: "clamp" });
  return (
    <div
      style={{
        position: "absolute",
        left: 70,
        right: 70,
        bottom: 230,
        textAlign: "center",
        fontFamily: SANS,
        fontWeight: 800,
        fontSize: 62,
        lineHeight: 1.12,
        textTransform: "uppercase",
        transform: `translateY(${interpolate(enter, [0, 1], [26, 0])}px)`,
        opacity: enter,
      }}
    >
      {page.words.map((w, i) => {
        const active = nowMs >= w.startMs && nowMs <= w.endMs + 60;
        return (
          <span
            key={i}
            style={{
              color: active ? "#FFE500" : "#ffffff",
              WebkitTextStroke: "10px black",
              paintOrder: "stroke fill",
              margin: "0 8px",
              display: "inline-block",
              transform: active ? "scale(1.06)" : "scale(1)",
            }}
          >
            {w.text}
          </span>
        );
      })}
    </div>
  );
};

// ---------- scene visuals ----------

const Center: React.FC<{ children: React.ReactNode }> = ({ children }) => (
  <AbsoluteFill style={{ justifyContent: "center", alignItems: "center", padding: "0 90px" }}>{children}</AbsoluteFill>
);

const TitleCard: React.FC<{ headline: string; sub: string }> = ({ headline, sub }) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const s = spring({ frame, fps, config: { damping: 14, mass: 0.7 } });
  return (
    <Center>
      <div style={{ textAlign: "center", transform: `scale(${interpolate(s, [0, 1], [0.8, 1])})`, opacity: s }}>
        <div style={{ fontFamily: MONO, color: C.accent, fontSize: 34, letterSpacing: 6, marginBottom: 28 }}>JS · 50 SECONDS</div>
        <div style={{ fontFamily: SANS, color: C.text, fontSize: 168, fontWeight: 900, lineHeight: 0.98, letterSpacing: -2 }}>
          {headline}
        </div>
        <div style={{ fontFamily: SANS, color: C.dim, fontSize: 46, marginTop: 30, fontWeight: 500 }}>{sub}</div>
      </div>
    </Center>
  );
};

const ConceptCard: React.FC<{ lead: string; highlight: string; tail: string }> = ({ lead, highlight, tail }) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const s = spring({ frame, fps, config: { damping: 16 } });
  const hl = spring({ frame: frame - 8, fps, config: { damping: 18 } });
  return (
    <Center>
      <div style={{ textAlign: "center", fontFamily: SANS, fontSize: 84, fontWeight: 800, lineHeight: 1.18, color: C.text, opacity: s }}>
        {lead}{" "}
        <span
          style={{
            color: C.bg0,
            background: `linear-gradient(90deg, ${C.accent}, ${C.accent2})`,
            padding: "6px 22px",
            borderRadius: 18,
            boxDecorationBreak: "clone",
            WebkitBoxDecorationBreak: "clone",
            transform: `scale(${interpolate(hl, [0, 1], [0.9, 1])})`,
            display: "inline-block",
          }}
        >
          {highlight}
        </span>{" "}
        {tail}
      </div>
    </Center>
  );
};

type Tok = [string, keyof typeof C | string];
const CodeCard: React.FC<{ caption: string; lines: Tok[][]; revealByLine: boolean; highlightLines?: number[] }> = ({
  caption,
  lines,
  highlightLines,
}) => {
  const frame = useCurrentFrame();
  const { fps, durationInFrames } = useVideoConfig();
  const s = spring({ frame, fps, config: { damping: 16 } });
  const perLine = (durationInFrames * 0.55) / Math.max(lines.length, 1);
  return (
    <Center>
      <div
        style={{
          width: "100%",
          background: C.panel,
          border: `1px solid ${C.border}`,
          borderRadius: 28,
          padding: "34px 40px 42px",
          boxShadow: "0 40px 120px rgba(0,0,0,0.5)",
          backdropFilter: "blur(6px)",
          transform: `translateY(${interpolate(s, [0, 1], [40, 0])}px)`,
          opacity: s,
        }}
      >
        <div style={{ display: "flex", alignItems: "center", gap: 12, marginBottom: 30 }}>
          <span style={{ width: 18, height: 18, borderRadius: 99, background: "#ff5f56" }} />
          <span style={{ width: 18, height: 18, borderRadius: 99, background: "#ffbd2e" }} />
          <span style={{ width: 18, height: 18, borderRadius: 99, background: "#27c93f" }} />
          <span style={{ marginLeft: 16, fontFamily: MONO, color: C.dim, fontSize: 30 }}>{caption}</span>
        </div>
        <pre style={{ margin: 0, fontFamily: MONO, fontSize: 46, lineHeight: 1.52 }}>
          {lines.map((toks, li) => {
            const appear = interpolate(frame, [li * perLine, li * perLine + 8], [0, 1], { extrapolateLeft: "clamp", extrapolateRight: "clamp" });
            const isHl = highlightLines?.includes(li);
            return (
              <div
                key={li}
                style={{
                  opacity: appear,
                  transform: `translateX(${interpolate(appear, [0, 1], [-18, 0])}px)`,
                  background: isHl ? "rgba(56,189,248,0.12)" : "transparent",
                  borderLeft: isHl ? `4px solid ${C.accent}` : "4px solid transparent",
                  paddingLeft: 14,
                  borderRadius: 6,
                }}
              >
                {toks.map((t, ti) => (
                  <span key={ti} style={{ color: (C as any)[t[1]] ?? C.plain, whiteSpace: "pre" }}>
                    {t[0]}
                  </span>
                ))}
                {toks.length === 0 ? " " : null}
              </div>
            );
          })}
        </pre>
      </div>
    </Center>
  );
};

const CounterBox: React.FC<{ name: string; counts: number[]; color: string; delay: number }> = ({ name, counts, color, delay }) => {
  const frame = useCurrentFrame();
  const { fps, durationInFrames } = useVideoConfig();
  const enter = spring({ frame: frame - delay, fps, config: { damping: 16 } });
  // step through counts across the scene
  const prog = interpolate(frame, [durationInFrames * 0.15, durationInFrames * 0.8], [0, counts.length], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });
  const idx = Math.min(counts.length - 1, Math.floor(prog));
  const value = counts[idx];
  const pop = spring({ frame: frame - (durationInFrames * 0.15 + idx * ((durationInFrames * 0.65) / counts.length)), fps, config: { damping: 12, mass: 0.5 } });
  return (
    <div
      style={{
        flex: 1,
        background: C.panel,
        border: `1px solid ${color}55`,
        borderRadius: 28,
        padding: "40px 30px",
        textAlign: "center",
        opacity: enter,
        transform: `translateY(${interpolate(enter, [0, 1], [40, 0])}px)`,
        boxShadow: `0 30px 80px rgba(0,0,0,0.45)`,
      }}
    >
      <div style={{ fontFamily: MONO, fontSize: 40, color: C.dim, marginBottom: 18 }}>{name}</div>
      <div style={{ fontFamily: SANS, fontWeight: 900, fontSize: 220, color, lineHeight: 1, transform: `scale(${interpolate(pop, [0, 1], [0.7, 1])})` }}>
        {value}
      </div>
      <div style={{ fontFamily: MONO, fontSize: 30, color: C.dim, marginTop: 18 }}>
        own scope
      </div>
    </div>
  );
};

const ScopeViz: React.FC<{ aCounts: number[]; bCounts: number[] }> = ({ aCounts, bCounts }) => (
  <Center>
    <div style={{ width: "100%" }}>
      <div style={{ display: "flex", gap: 40 }}>
        <CounterBox name="a()" counts={aCounts} color={C.accent} delay={0} />
        <CounterBox name="b()" counts={bCounts} color={C.accent2} delay={6} />
      </div>
    </div>
  </Center>
);

const RecapCard: React.FC<{ line1: string; line2: string }> = ({ line1, line2 }) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const s = spring({ frame, fps, config: { damping: 16 } });
  const s2 = spring({ frame: frame - 12, fps, config: { damping: 16 } });
  return (
    <Center>
      <div style={{ textAlign: "center", fontFamily: MONO, fontWeight: 700 }}>
        <div style={{ fontSize: 40, letterSpacing: 6, color: C.accent, marginBottom: 40, fontFamily: SANS }}>RECAP</div>
        <div style={{ fontSize: 74, color: C.text, opacity: s, transform: `translateY(${interpolate(s, [0, 1], [24, 0])}px)` }}>{line1}</div>
        <div style={{ fontSize: 74, color: C.accent2, opacity: s2, marginTop: 18, transform: `translateY(${interpolate(s2, [0, 1], [24, 0])}px)` }}>{line2}</div>
      </div>
    </Center>
  );
};

const CtaCard: React.FC<{ handle: string; sub: string }> = ({ handle, sub }) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const s = spring({ frame, fps, config: { damping: 13, mass: 0.6 } });
  const pulse = 1 + Math.sin(frame / 6) * 0.02;
  return (
    <Center>
      <div style={{ textAlign: "center", transform: `scale(${interpolate(s, [0, 1], [0.7, 1]) * pulse})`, opacity: s }}>
        <div style={{ fontFamily: SANS, fontSize: 64, fontWeight: 700, color: C.text, marginBottom: 26 }}>Follow</div>
        <div
          style={{
            fontFamily: SANS,
            fontSize: 110,
            fontWeight: 900,
            background: `linear-gradient(90deg, ${C.accent}, ${C.accent2})`,
            WebkitBackgroundClip: "text",
            backgroundClip: "text",
            color: "transparent",
          }}
        >
          {handle}
        </div>
        <div style={{ fontFamily: MONO, fontSize: 40, color: C.dim, marginTop: 28 }}>{sub}</div>
      </div>
    </Center>
  );
};

const renderVisual = (v: any) => {
  switch (v.kind) {
    case "title": return <TitleCard headline={v.headline} sub={v.sub} />;
    case "concept": return <ConceptCard lead={v.lead} highlight={v.highlight} tail={v.tail} />;
    case "code": return <CodeCard caption={v.caption} lines={v.lines} revealByLine={v.revealByLine} highlightLines={v.highlightLines} />;
    case "scope": return <ScopeViz aCounts={v.aCounts} bCounts={v.bCounts} />;
    case "recap": return <RecapCard line1={v.line1} line2={v.line2} />;
    case "cta": return <CtaCard handle={v.handle} sub={v.sub} />;
    default: return null;
  }
};

// ---------- root ----------

export const LessonReel: React.FC<{ lesson: any }> = ({ lesson }) => {
  const { fps, durationInFrames } = useVideoConfig();
  if (!lesson) return <AbsoluteFill style={{ background: C.bg0 }} />;
  const lastIdx = lesson.scenes.length - 1;
  return (
    <AbsoluteFill style={{ background: C.bg0 }}>
      <Background />
      <Audio src={staticFile(`content/${lesson.slug}/${lesson.audioUrl}`)} />
      <Brand handle={lesson.handle} title={lesson.title} />
      {lesson.scenes.map((scene: any, i: number) => {
        const from = ms2f(scene.startMs, fps);
        // Hold the final scene (CTA) to the end of the composition — no dead tail.
        const end = i === lastIdx ? durationInFrames : ms2f(scene.endMs, fps);
        const dur = Math.max(1, end - from);
        return (
          <Sequence key={i} from={from} durationInFrames={dur}>
            {renderVisual(scene.visual)}
          </Sequence>
        );
      })}
      <Captions capGroups={lesson.scenes.map((s: any) => s.captions)} />
      <Progress totalMs={lesson.totalMs} />
    </AbsoluteFill>
  );
};
