import { z } from "zod";
import {
  AbsoluteFill, Audio, Img, Sequence, interpolate, spring, staticFile,
  useCurrentFrame, useVideoConfig,
} from "remotion";
import { loadFont as loadInter } from "@remotion/google-fonts/Inter";
import { loadFont as loadMono } from "@remotion/google-fonts/RobotoMono";
import { Particles } from "./Character";

const { fontFamily: SANS } = loadInter("normal", { weights: ["500", "700", "800", "900"], subsets: ["latin"], ignoreTooManyRequestsWarning: true });
const { fontFamily: MONO } = loadMono("normal", { weights: ["400", "700"], subsets: ["latin"], ignoreTooManyRequestsWarning: true });

export const lessonAISchema = z.object({ lesson: z.any().nullable() });

const C = {
  bg0: "#0a0e1a", bg1: "#141d36", panel: "rgba(17,26,46,0.78)", border: "rgba(120,160,255,0.20)",
  accent: "#38bdf8", accent2: "#34d399", text: "#e7ecf5", dim: "#8a97b5",
  kw: "#c792ea", ret: "#c792ea", fn: "#82aaff", var: "#f78c6c", num: "#f78c6c", punct: "#89ddff", comment: "#5c6a86", plain: "#d6deeb",
};
const ms2f = (ms: number, fps: number) => Math.round((ms / 1000) * fps);
const asset = (slug: string, name: string) => staticFile(`content/${slug}/assets/${name}`);

// floating character image with spring entrance + idle bob
const FloatImg: React.FC<{ src: string; size: number; delay?: number; seed?: number; flip?: boolean; hue?: number; bounce?: boolean }> = ({ src, size, delay = 0, seed = 0, flip = false, hue = 0, bounce = false }) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const s = spring({ frame: frame - delay, fps, config: { damping: 14, mass: 0.8 } });
  const bob = Math.sin((frame + seed * 20) / 16) * 10 - (bounce ? Math.abs(Math.sin(frame / 7)) * 22 : 0);
  const tilt = Math.sin((frame + seed * 15) / 40) * 2;
  return (
    <div style={{ transform: `translateY(${interpolate(s, [0, 1], [60, bob])}px) scale(${interpolate(s, [0, 1], [0.7, 1])}) rotate(${tilt}deg) scaleX(${flip ? -1 : 1})`, opacity: s, filter: `drop-shadow(0 24px 40px rgba(0,0,0,0.5)) hue-rotate(${hue}deg)` }}>
      <Img src={src} style={{ width: size, height: size * 1.29, objectFit: "contain" }} />
    </div>
  );
};

const Background: React.FC = () => {
  const frame = useCurrentFrame();
  return (
    <AbsoluteFill>
      <AbsoluteFill style={{ background: `radial-gradient(130% 100% at 50% -10%, ${C.bg1} 0%, ${C.bg0} 68%)` }} />
      <AbsoluteFill style={{ transform: `translateY(${Math.sin(frame / 90) * 12}px)`, maskImage: "radial-gradient(75% 60% at 50% 42%, black, transparent 92%)" }}><Particles accent={C.accent} /></AbsoluteFill>
      <AbsoluteFill style={{ background: `radial-gradient(55% 26% at 50% 4%, ${C.accent}22 0%, transparent 70%)` }} />
    </AbsoluteFill>
  );
};
const Brand: React.FC<{ handle: string; title: string }> = ({ handle, title }) => (
  <div style={{ position: "absolute", top: 70, left: 0, right: 0, display: "flex", justifyContent: "center", gap: 14, fontFamily: SANS, alignItems: "center" }}>
    <div style={{ width: 14, height: 14, borderRadius: 4, background: C.accent2, boxShadow: `0 0 18px ${C.accent2}` }} />
    <span style={{ color: C.text, fontSize: 30, fontWeight: 700 }}>{title}</span>
    <span style={{ color: C.dim, fontSize: 28 }}>· {handle}</span>
  </div>
);
const Progress: React.FC<{ totalMs: number }> = ({ totalMs }) => {
  const frame = useCurrentFrame(); const { fps } = useVideoConfig();
  const pct = Math.min(1, (frame / fps) * 1000 / totalMs);
  return (<div style={{ position: "absolute", left: 80, right: 80, bottom: 70, height: 8, borderRadius: 99, background: "rgba(255,255,255,0.10)" }}><div style={{ width: `${pct * 100}%`, height: "100%", borderRadius: 99, background: `linear-gradient(90deg, ${C.accent}, ${C.accent2})`, boxShadow: `0 0 16px ${C.accent}88` }} /></div>);
};

type Cap = { text: string; startMs: number; endMs: number };
const buildPages = (caps: Cap[], maxWords = 4, maxMs = 1500) => {
  const pages: { words: Cap[]; startMs: number; endMs: number }[] = []; let cur: Cap[] = [];
  for (const w of caps) { if (cur.length && (cur.length >= maxWords || w.endMs - cur[0].startMs > maxMs)) { pages.push({ words: cur, startMs: cur[0].startMs, endMs: cur[cur.length - 1].endMs }); cur = []; } cur.push(w); }
  if (cur.length) pages.push({ words: cur, startMs: cur[0].startMs, endMs: cur[cur.length - 1].endMs }); return pages;
};
const Captions: React.FC<{ capGroups: Cap[][] }> = ({ capGroups }) => {
  const frame = useCurrentFrame(); const { fps } = useVideoConfig(); const nowMs = (frame / fps) * 1000;
  const pages = capGroups.flatMap((g) => buildPages(g));
  const page = pages.find((p) => nowMs >= p.startMs && nowMs <= p.endMs + 220) ?? null;
  if (!page) return null;
  const enter = interpolate(nowMs, [page.startMs, page.startMs + 130], [0, 1], { extrapolateLeft: "clamp", extrapolateRight: "clamp" });
  return (
    <div style={{ position: "absolute", left: 70, right: 70, bottom: 180, textAlign: "center", fontFamily: SANS, fontWeight: 800, fontSize: 58, lineHeight: 1.12, textTransform: "uppercase", transform: `translateY(${interpolate(enter, [0, 1], [26, 0])}px)`, opacity: enter }}>
      {page.words.map((w, i) => {
        const active = nowMs >= w.startMs && nowMs <= w.endMs + 60;
        return (<span key={i} style={{ color: active ? "#FFE500" : "#fff", WebkitTextStroke: "10px black", paintOrder: "stroke fill", margin: "0 8px", display: "inline-block", transform: active ? "scale(1.06)" : "scale(1)" }}>{w.text}</span>);
      })}
    </div>
  );
};

const Center: React.FC<{ children: React.ReactNode; top?: number }> = ({ children, top }) => (
  <AbsoluteFill style={{ justifyContent: top != null ? "flex-start" : "center", alignItems: "center", padding: "0 80px", paddingTop: top }}>{children}</AbsoluteFill>
);

const CountBadge: React.FC<{ value: number; color: string }> = ({ value, color }) => {
  const frame = useCurrentFrame(); const { fps } = useVideoConfig();
  const pop = spring({ frame: frame % 999, fps, config: { damping: 10 } });
  return (
    <div style={{ background: C.bg0, border: `3px solid ${color}`, borderRadius: 20, padding: "10px 26px", fontFamily: MONO, fontWeight: 700, fontSize: 64, color, boxShadow: `0 0 28px ${color}88`, transform: `scale(${interpolate(pop, [0, 1], [0.85, 1])})` }}>
      {value}
    </div>
  );
};

// ---- scenes ----
const AiTitle: React.FC<{ slug: string; headline: string; sub: string }> = ({ slug, headline, sub }) => {
  const frame = useCurrentFrame(); const { fps } = useVideoConfig();
  const s = spring({ frame, fps, config: { damping: 14 } });
  return (
    <Center top={280}>
      <div style={{ textAlign: "center", opacity: s, transform: `scale(${interpolate(s, [0, 1], [0.82, 1])})` }}>
        <div style={{ fontFamily: MONO, color: C.accent, fontSize: 32, letterSpacing: 6, marginBottom: 16 }}>JS · 50 SECONDS</div>
        <div style={{ fontFamily: SANS, color: C.text, fontSize: 150, fontWeight: 900, lineHeight: 0.98, letterSpacing: -2 }}>{headline}</div>
        <div style={{ fontFamily: SANS, color: C.dim, fontSize: 42, marginTop: 18 }}>{sub}</div>
      </div>
      <div style={{ marginTop: 30 }}><FloatImg src={asset(slug, "pip.png")} size={560} delay={6} /></div>
    </Center>
  );
};
const AiConcept: React.FC<{ slug: string; lead: string; highlight: string; tail: string }> = ({ slug, lead, highlight, tail }) => {
  const frame = useCurrentFrame(); const { fps } = useVideoConfig();
  const s = spring({ frame, fps, config: { damping: 16 } }); const hl = spring({ frame: frame - 8, fps, config: { damping: 18 } });
  return (
    <Center>
      <div style={{ position: "absolute", right: 30, top: 300 }}><FloatImg src={asset(slug, "pip.png")} size={280} seed={3} flip /></div>
      <div style={{ textAlign: "center", fontFamily: SANS, fontSize: 80, fontWeight: 800, lineHeight: 1.2, color: C.text, opacity: s, maxWidth: 840 }}>
        {lead}{" "}<span style={{ color: C.bg0, background: `linear-gradient(90deg, ${C.accent}, ${C.accent2})`, padding: "6px 20px", borderRadius: 16, display: "inline-block", transform: `scale(${interpolate(hl, [0, 1], [0.9, 1])})` }}>{highlight}</span>{" "}{tail}
      </div>
    </Center>
  );
};
type Tok = [string, string];
const AiCode: React.FC<{ slug: string; caption: string; lines: Tok[][]; highlightLines?: number[] }> = ({ slug, caption, lines, highlightLines }) => {
  const frame = useCurrentFrame(); const { fps, durationInFrames } = useVideoConfig();
  const s = spring({ frame, fps, config: { damping: 16 } });
  const perLine = (durationInFrames * 0.5) / Math.max(lines.length, 1);
  return (
    <Center>
      <div style={{ width: "90%", background: C.panel, border: `1px solid ${C.border}`, borderRadius: 26, padding: "30px 36px 38px", boxShadow: "0 40px 120px rgba(0,0,0,0.5)", transform: `translateY(${interpolate(s, [0, 1], [40, 0])}px)`, opacity: s }}>
        <div style={{ display: "flex", alignItems: "center", gap: 12, marginBottom: 24 }}>
          <span style={{ width: 16, height: 16, borderRadius: 99, background: "#ff5f56" }} /><span style={{ width: 16, height: 16, borderRadius: 99, background: "#ffbd2e" }} /><span style={{ width: 16, height: 16, borderRadius: 99, background: "#27c93f" }} />
          <span style={{ marginLeft: 14, fontFamily: MONO, color: C.dim, fontSize: 28 }}>{caption}</span>
        </div>
        <pre style={{ margin: 0, fontFamily: MONO, fontSize: 42, lineHeight: 1.5 }}>
          {lines.map((toks, li) => {
            const appear = interpolate(frame, [li * perLine, li * perLine + 8], [0, 1], { extrapolateLeft: "clamp", extrapolateRight: "clamp" });
            const isHl = highlightLines?.includes(li);
            return (<div key={li} style={{ opacity: appear, transform: `translateX(${interpolate(appear, [0, 1], [-18, 0])}px)`, background: isHl ? "rgba(56,189,248,0.12)" : "transparent", borderLeft: isHl ? `4px solid ${C.accent}` : "4px solid transparent", paddingLeft: 12, borderRadius: 6 }}>{toks.map((t, ti) => (<span key={ti} style={{ color: (C as any)[t[1]] ?? C.plain, whiteSpace: "pre" }}>{t[0]}</span>))}{toks.length === 0 ? " " : null}</div>);
          })}
        </pre>
      </div>
      <div style={{ position: "absolute", bottom: 230, right: 40 }}><FloatImg src={asset(slug, "pip.png")} size={210} seed={5} flip /></div>
    </Center>
  );
};
const AiBuddy: React.FC<{ slug: string; name: string; counts: number[]; color: string; seed: number; hue?: number; flip?: boolean }> = ({ slug, name, counts, color, seed, hue = 0, flip = false }) => {
  const frame = useCurrentFrame(); const { fps, durationInFrames } = useVideoConfig();
  const prog = interpolate(frame, [durationInFrames * 0.2, durationInFrames * 0.82], [0, counts.length], { extrapolateLeft: "clamp", extrapolateRight: "clamp" });
  const idx = Math.min(counts.length - 1, Math.floor(prog));
  return (
    <div style={{ flex: 1, display: "flex", flexDirection: "column", alignItems: "center", gap: 6 }}>
      <div style={{ fontFamily: MONO, fontSize: 38, color }}>{name}</div>
      <CountBadge value={counts[idx]} color={color} />
      <FloatImg src={asset(slug, "pip.png")} size={300} delay={seed * 3} seed={seed} hue={hue} flip={flip} />
    </div>
  );
};
const AiScope: React.FC<{ slug: string; aCounts: number[]; bCounts: number[] }> = ({ slug, aCounts, bCounts }) => (
  <Center>
    <div style={{ width: "100%", display: "flex", gap: 20, justifyContent: "center", alignItems: "flex-start" }}>
      {/* same mascot twice = two instances of the same makeCounter factory */}
      <AiBuddy slug={slug} name="a()" counts={aCounts} color={C.accent} seed={1} />
      <AiBuddy slug={slug} name="b()" counts={bCounts} color={C.accent2} seed={4} hue={-55} flip />
    </div>
  </Center>
);
const AiRecap: React.FC<{ slug: string; line1: string; line2: string }> = ({ slug, line1, line2 }) => {
  const frame = useCurrentFrame(); const { fps } = useVideoConfig();
  const s = spring({ frame, fps, config: { damping: 16 } }); const s2 = spring({ frame: frame - 12, fps, config: { damping: 16 } });
  return (
    <Center>
      <div style={{ position: "absolute", left: 20, top: 360 }}><FloatImg src={asset(slug, "pip.png")} size={260} seed={2} bounce /></div>
      <div style={{ textAlign: "center", fontFamily: MONO, fontWeight: 700 }}>
        <div style={{ fontSize: 38, letterSpacing: 6, color: C.accent, marginBottom: 36, fontFamily: SANS }}>RECAP</div>
        <div style={{ fontSize: 68, color: C.text, opacity: s, transform: `translateY(${interpolate(s, [0, 1], [24, 0])}px)` }}>{line1}</div>
        <div style={{ fontSize: 68, color: C.accent2, opacity: s2, marginTop: 16, transform: `translateY(${interpolate(s2, [0, 1], [24, 0])}px)` }}>{line2}</div>
      </div>
    </Center>
  );
};
const AiCta: React.FC<{ slug: string; handle: string; sub: string }> = ({ slug, handle, sub }) => {
  const frame = useCurrentFrame(); const { fps } = useVideoConfig();
  const s = spring({ frame, fps, config: { damping: 13, mass: 0.6 } }); const pulse = 1 + Math.sin(frame / 6) * 0.02;
  return (
    <Center top={300}>
      <div style={{ textAlign: "center", transform: `scale(${interpolate(s, [0, 1], [0.7, 1]) * pulse})`, opacity: s }}>
        <div style={{ fontFamily: SANS, fontSize: 58, fontWeight: 700, color: C.text, marginBottom: 18 }}>Follow</div>
        <div style={{ fontFamily: SANS, fontSize: 104, fontWeight: 900, background: `linear-gradient(90deg, ${C.accent}, ${C.accent2})`, WebkitBackgroundClip: "text", backgroundClip: "text", color: "transparent" }}>{handle}</div>
        <div style={{ fontFamily: MONO, fontSize: 38, color: C.dim, marginTop: 18 }}>{sub}</div>
      </div>
      <div style={{ marginTop: 20 }}><FloatImg src={asset(slug, "pip.png")} size={460} delay={6} bounce /></div>
    </Center>
  );
};

const render = (slug: string, v: any) => {
  switch (v.kind) {
    case "title": return <AiTitle slug={slug} headline={v.headline} sub={v.sub} />;
    case "concept": return <AiConcept slug={slug} lead={v.lead} highlight={v.highlight} tail={v.tail} />;
    case "code": return <AiCode slug={slug} caption={v.caption} lines={v.lines} highlightLines={v.highlightLines} />;
    case "scope": return <AiScope slug={slug} aCounts={v.aCounts} bCounts={v.bCounts} />;
    case "recap": return <AiRecap slug={slug} line1={v.line1} line2={v.line2} />;
    case "cta": return <AiCta slug={slug} handle={v.handle} sub={v.sub} />;
    default: return null;
  }
};

export const LessonReelAI: React.FC<{ lesson: any }> = ({ lesson }) => {
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
        return (<Sequence key={i} from={from} durationInFrames={Math.max(1, end - from)}>{render(lesson.slug, scene.visual)}</Sequence>);
      })}
      <Captions capGroups={lesson.scenes.map((s: any) => s.captions)} />
      <Progress totalMs={lesson.totalMs} />
    </AbsoluteFill>
  );
};
