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
import { Character, Particles } from "./Character";

const { fontFamily: SANS } = loadInter("normal", { weights: ["500", "700", "800", "900"], subsets: ["latin"], ignoreTooManyRequestsWarning: true });
const { fontFamily: MONO } = loadMono("normal", { weights: ["400", "700"], subsets: ["latin"], ignoreTooManyRequestsWarning: true });

export const lessonProSchema = z.object({ lesson: z.any().nullable() });

const C = {
  bg0: "#0a0e1a", bg1: "#141d36", panel: "rgba(17,26,46,0.78)", border: "rgba(120,160,255,0.20)",
  accent: "#38bdf8", accent2: "#34d399", text: "#e7ecf5", dim: "#8a97b5",
  kw: "#c792ea", ret: "#c792ea", fn: "#82aaff", var: "#f78c6c", num: "#f78c6c",
  punct: "#89ddff", comment: "#5c6a86", plain: "#d6deeb",
};
const ms2f = (ms: number, fps: number) => Math.round((ms / 1000) * fps);

const Background: React.FC = () => {
  const frame = useCurrentFrame();
  return (
    <AbsoluteFill>
      <AbsoluteFill style={{ background: `radial-gradient(130% 100% at 50% -10%, ${C.bg1} 0%, ${C.bg0} 68%)` }} />
      <AbsoluteFill style={{ transform: `translateY(${Math.sin(frame / 90) * 12}px)`, maskImage: "radial-gradient(75% 60% at 50% 42%, black, transparent 92%)" }}>
        <Particles accent={C.accent} />
      </AbsoluteFill>
      <AbsoluteFill style={{ background: `radial-gradient(55% 26% at 50% 4%, ${C.accent}22 0%, transparent 70%)` }} />
    </AbsoluteFill>
  );
};

const Brand: React.FC<{ handle: string; title: string }> = ({ handle, title }) => (
  <div style={{ position: "absolute", top: 70, left: 0, right: 0, display: "flex", justifyContent: "center", alignItems: "center", gap: 14, fontFamily: SANS }}>
    <div style={{ width: 14, height: 14, borderRadius: 4, background: C.accent2, boxShadow: `0 0 18px ${C.accent2}` }} />
    <span style={{ color: C.text, fontSize: 30, fontWeight: 700 }}>{title}</span>
    <span style={{ color: C.dim, fontSize: 28 }}>· {handle}</span>
  </div>
);

const Progress: React.FC<{ totalMs: number }> = ({ totalMs }) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const pct = Math.min(1, (frame / fps) * 1000 / totalMs);
  return (
    <div style={{ position: "absolute", left: 80, right: 80, bottom: 70, height: 8, borderRadius: 99, background: "rgba(255,255,255,0.10)" }}>
      <div style={{ width: `${pct * 100}%`, height: "100%", borderRadius: 99, background: `linear-gradient(90deg, ${C.accent}, ${C.accent2})`, boxShadow: `0 0 16px ${C.accent}88` }} />
    </div>
  );
};

// ---- captions ----
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
  const pages = capGroups.flatMap((g) => buildPages(g));
  const page = pages.find((p) => nowMs >= p.startMs && nowMs <= p.endMs + 220) ?? null;
  if (!page) return null;
  const enter = interpolate(nowMs, [page.startMs, page.startMs + 130], [0, 1], { extrapolateLeft: "clamp", extrapolateRight: "clamp" });
  return (
    <div style={{ position: "absolute", left: 70, right: 70, bottom: 200, textAlign: "center", fontFamily: SANS, fontWeight: 800, fontSize: 60, lineHeight: 1.12, textTransform: "uppercase", transform: `translateY(${interpolate(enter, [0, 1], [26, 0])}px)`, opacity: enter }}>
      {page.words.map((w, i) => {
        const active = nowMs >= w.startMs && nowMs <= w.endMs + 60;
        return (
          <span key={i} style={{ color: active ? "#FFE500" : "#fff", WebkitTextStroke: "10px black", paintOrder: "stroke fill", margin: "0 8px", display: "inline-block", transform: active ? "scale(1.06)" : "scale(1)" }}>
            {w.text}
          </span>
        );
      })}
    </div>
  );
};

const Center: React.FC<{ children: React.ReactNode; top?: number }> = ({ children, top }) => (
  <AbsoluteFill style={{ justifyContent: top != null ? "flex-start" : "center", alignItems: "center", padding: "0 80px", paddingTop: top }}>{children}</AbsoluteFill>
);

// ---- scenes ----
const TitleScene: React.FC<{ headline: string; sub: string }> = ({ headline, sub }) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const s = spring({ frame, fps, config: { damping: 13, mass: 0.7 } });
  return (
    <Center top={300}>
      <div style={{ textAlign: "center", opacity: s, transform: `scale(${interpolate(s, [0, 1], [0.8, 1])})` }}>
        <div style={{ fontFamily: MONO, color: C.accent, fontSize: 32, letterSpacing: 6, marginBottom: 18 }}>JS · 50 SECONDS</div>
        <div style={{ fontFamily: SANS, color: C.text, fontSize: 150, fontWeight: 900, lineHeight: 0.98, letterSpacing: -2 }}>{headline}</div>
        <div style={{ fontFamily: SANS, color: C.dim, fontSize: 42, marginTop: 22 }}>{sub}</div>
      </div>
      <div style={{ marginTop: 80 }}>
        <Character pose="wave" eyeDir={[0, 0.2]} size={460} />
      </div>
    </Center>
  );
};

const ConceptScene: React.FC<{ lead: string; highlight: string; tail: string }> = ({ lead, highlight, tail }) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const s = spring({ frame, fps, config: { damping: 16 } });
  const hl = spring({ frame: frame - 8, fps, config: { damping: 18 } });
  return (
    <Center>
      <div style={{ position: "absolute", right: 70, top: 320 }}>
        <Character pose="point" eyeDir={[-0.6, 0]} color={C.accent2} glow={C.accent2} size={300} seed={3} />
      </div>
      <div style={{ textAlign: "center", fontFamily: SANS, fontSize: 80, fontWeight: 800, lineHeight: 1.2, color: C.text, opacity: s, maxWidth: 820 }}>
        {lead}{" "}
        <span style={{ color: C.bg0, background: `linear-gradient(90deg, ${C.accent}, ${C.accent2})`, padding: "6px 20px", borderRadius: 16, WebkitBoxDecorationBreak: "clone", boxDecorationBreak: "clone", display: "inline-block", transform: `scale(${interpolate(hl, [0, 1], [0.9, 1])})` }}>{highlight}</span>{" "}
        {tail}
      </div>
    </Center>
  );
};

type Tok = [string, string];
const CodeScene: React.FC<{ caption: string; lines: Tok[][]; highlightLines?: number[] }> = ({ caption, lines, highlightLines }) => {
  const frame = useCurrentFrame();
  const { fps, durationInFrames } = useVideoConfig();
  const s = spring({ frame, fps, config: { damping: 16 } });
  const perLine = (durationInFrames * 0.5) / Math.max(lines.length, 1);
  return (
    <Center>
      <div style={{ width: "92%", background: C.panel, border: `1px solid ${C.border}`, borderRadius: 26, padding: "30px 36px 38px", boxShadow: "0 40px 120px rgba(0,0,0,0.5)", transform: `translateY(${interpolate(s, [0, 1], [40, 0])}px)`, opacity: s }}>
        <div style={{ display: "flex", alignItems: "center", gap: 12, marginBottom: 24 }}>
          <span style={{ width: 16, height: 16, borderRadius: 99, background: "#ff5f56" }} />
          <span style={{ width: 16, height: 16, borderRadius: 99, background: "#ffbd2e" }} />
          <span style={{ width: 16, height: 16, borderRadius: 99, background: "#27c93f" }} />
          <span style={{ marginLeft: 14, fontFamily: MONO, color: C.dim, fontSize: 28 }}>{caption}</span>
        </div>
        <pre style={{ margin: 0, fontFamily: MONO, fontSize: 42, lineHeight: 1.5 }}>
          {lines.map((toks, li) => {
            const appear = interpolate(frame, [li * perLine, li * perLine + 8], [0, 1], { extrapolateLeft: "clamp", extrapolateRight: "clamp" });
            const isHl = highlightLines?.includes(li);
            return (
              <div key={li} style={{ opacity: appear, transform: `translateX(${interpolate(appear, [0, 1], [-18, 0])}px)`, background: isHl ? "rgba(56,189,248,0.12)" : "transparent", borderLeft: isHl ? `4px solid ${C.accent}` : "4px solid transparent", paddingLeft: 12, borderRadius: 6 }}>
                {toks.map((t, ti) => (<span key={ti} style={{ color: (C as any)[t[1]] ?? C.plain, whiteSpace: "pre" }}>{t[0]}</span>))}
                {toks.length === 0 ? " " : null}
              </div>
            );
          })}
        </pre>
      </div>
      <div style={{ position: "absolute", bottom: 250, right: 90 }}>
        <Character pose="point" eyeDir={[-0.4, -0.4]} size={220} seed={5} />
      </div>
    </Center>
  );
};

const Buddy: React.FC<{ name: string; counts: number[]; color: string; seed: number }> = ({ name, counts, color, seed }) => {
  const frame = useCurrentFrame();
  const { fps, durationInFrames } = useVideoConfig();
  const enter = spring({ frame: frame - seed * 4, fps, config: { damping: 15 } });
  const prog = interpolate(frame, [durationInFrames * 0.2, durationInFrames * 0.82], [0, counts.length], { extrapolateLeft: "clamp", extrapolateRight: "clamp" });
  const idx = Math.min(counts.length - 1, Math.floor(prog));
  return (
    <div style={{ flex: 1, display: "flex", flexDirection: "column", alignItems: "center", opacity: enter, transform: `translateY(${interpolate(enter, [0, 1], [50, 0])}px)` }}>
      <div style={{ fontFamily: MONO, fontSize: 40, color, marginBottom: 8 }}>{name}</div>
      <Character pose="hold" color={color} glow={color} size={320} seed={seed} holdValue={`count: ${counts[idx]}`} />
    </div>
  );
};
const ScopeScene: React.FC<{ aCounts: number[]; bCounts: number[] }> = ({ aCounts, bCounts }) => (
  <Center>
    <div style={{ width: "100%", display: "flex", gap: 30, justifyContent: "center" }}>
      <Buddy name="a()" counts={aCounts} color={C.accent} seed={1} />
      <Buddy name="b()" counts={bCounts} color={C.accent2} seed={6} />
    </div>
  </Center>
);

const RecapScene: React.FC<{ line1: string; line2: string }> = ({ line1, line2 }) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const s = spring({ frame, fps, config: { damping: 16 } });
  const s2 = spring({ frame: frame - 12, fps, config: { damping: 16 } });
  return (
    <Center>
      <div style={{ position: "absolute", left: 80, top: 360 }}>
        <Character pose="cheer" color={C.accent2} glow={C.accent2} size={260} seed={2} />
      </div>
      <div style={{ textAlign: "center", fontFamily: MONO, fontWeight: 700 }}>
        <div style={{ fontSize: 38, letterSpacing: 6, color: C.accent, marginBottom: 36, fontFamily: SANS }}>RECAP</div>
        <div style={{ fontSize: 70, color: C.text, opacity: s, transform: `translateY(${interpolate(s, [0, 1], [24, 0])}px)` }}>{line1}</div>
        <div style={{ fontSize: 70, color: C.accent2, opacity: s2, marginTop: 16, transform: `translateY(${interpolate(s2, [0, 1], [24, 0])}px)` }}>{line2}</div>
      </div>
    </Center>
  );
};

const CtaScene: React.FC<{ handle: string; sub: string }> = ({ handle, sub }) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const s = spring({ frame, fps, config: { damping: 13, mass: 0.6 } });
  const pulse = 1 + Math.sin(frame / 6) * 0.02;
  return (
    <Center top={360}>
      <div style={{ textAlign: "center", transform: `scale(${interpolate(s, [0, 1], [0.7, 1]) * pulse})`, opacity: s }}>
        <div style={{ fontFamily: SANS, fontSize: 60, fontWeight: 700, color: C.text, marginBottom: 22 }}>Follow</div>
        <div style={{ fontFamily: SANS, fontSize: 104, fontWeight: 900, background: `linear-gradient(90deg, ${C.accent}, ${C.accent2})`, WebkitBackgroundClip: "text", backgroundClip: "text", color: "transparent" }}>{handle}</div>
        <div style={{ fontFamily: MONO, fontSize: 38, color: C.dim, marginTop: 22 }}>{sub}</div>
      </div>
      <div style={{ marginTop: 50 }}>
        <Character pose="cheer" size={400} />
      </div>
    </Center>
  );
};

const renderVisual = (v: any) => {
  switch (v.kind) {
    case "title": return <TitleScene headline={v.headline} sub={v.sub} />;
    case "concept": return <ConceptScene lead={v.lead} highlight={v.highlight} tail={v.tail} />;
    case "code": return <CodeScene caption={v.caption} lines={v.lines} highlightLines={v.highlightLines} />;
    case "scope": return <ScopeScene aCounts={v.aCounts} bCounts={v.bCounts} />;
    case "recap": return <RecapScene line1={v.line1} line2={v.line2} />;
    case "cta": return <CtaScene handle={v.handle} sub={v.sub} />;
    default: return null;
  }
};

export const LessonReelPro: React.FC<{ lesson: any }> = ({ lesson }) => {
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
        const end = i === lastIdx ? durationInFrames : ms2f(scene.endMs, fps);
        return (
          <Sequence key={i} from={from} durationInFrames={Math.max(1, end - from)}>
            {renderVisual(scene.visual)}
          </Sequence>
        );
      })}
      <Captions capGroups={lesson.scenes.map((s: any) => s.captions)} />
      <Progress totalMs={lesson.totalMs} />
    </AbsoluteFill>
  );
};
