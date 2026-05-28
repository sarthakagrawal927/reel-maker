import { loadFont } from "@remotion/google-fonts/BreeSerif";
import { Audio } from "@remotion/media";
import { AbsoluteFill, Sequence, staticFile, useVideoConfig } from "remotion";
import { z } from "zod";
import { FPS, INTRO_DURATION } from "../lib/constants";
import { TimelineSchema, VideoStyleSchema } from "../lib/types";
import { calculateFrameTiming, getAudioPath } from "../lib/utils";
import { Background } from "./Background";
import Subtitle from "./Subtitle";

export const aiVideoSchema = z.object({
  timeline: TimelineSchema.nullable(),
});

const { fontFamily } = loadFont();

const DEFAULT_STYLE = VideoStyleSchema.parse({});

export const AIVideo: React.FC<z.infer<typeof aiVideoSchema>> = ({
  timeline,
}) => {
  if (!timeline) {
    throw new Error("Expected timeline to be fetched");
  }

  const { id, durationInFrames } = useVideoConfig();
  const style = timeline.style ?? DEFAULT_STYLE;

  return (
    <AbsoluteFill style={{ backgroundColor: "#05070b" }}>
      <Sequence durationInFrames={INTRO_DURATION}>
        <AbsoluteFill style={introStyle}>
          <div style={eyebrowStyle}>ACTIVE AI REEL</div>
          <div style={{ ...heroCardStyle, fontFamily }}>
            {timeline.shortTitle}
          </div>
          <div style={introSubStyle}>real product proof, not generic AI hype</div>
        </AbsoluteFill>
      </Sequence>

      {timeline.elements.map((element, index) => {
        const { startFrame, duration } = calculateFrameTiming(
          element.startMs,
          element.endMs,
          { includeIntro: index === 0 },
        );

        return (
          <Sequence
            key={`element-${index}`}
            from={startFrame}
            durationInFrames={duration}
            premountFor={3 * FPS}
          >
            <Background project={id} item={element} />
            <SceneOverlay
              index={index}
              title={timeline.shortTitle}
              caption={timeline.text[index]?.captions.map((caption) => caption.text).join("").trim() ?? ""}
            />
          </Sequence>
        );
      })}

      {timeline.text.map((element, index) => {
        const { startFrame, duration } = calculateFrameTiming(
          element.startMs,
          element.endMs,
          { addIntroOffset: true },
        );

        return (
          <Sequence
            key={`text-${index}`}
            from={startFrame}
            durationInFrames={duration}
          >
            <Subtitle
              captions={element.captions}
              sceneStartMs={element.startMs}
              style={style}
            />
          </Sequence>
        );
      })}

      {timeline.audio.map((element, index) => {
        const { startFrame, duration } = calculateFrameTiming(
          element.startMs,
          element.endMs,
          { addIntroOffset: true },
        );

        return (
          <Sequence
            key={`audio-${index}`}
            from={startFrame}
            durationInFrames={duration}
            premountFor={3 * FPS}
          >
            <Audio src={staticFile(getAudioPath(id, element.audioUrl))} />
          </Sequence>
        );
      })}

      {timeline.backgroundMusic && (
        <Sequence from={0} durationInFrames={durationInFrames}>
          <Audio
            src={staticFile(
              `content/${id}/${timeline.backgroundMusic.url}`,
            )}
            volume={timeline.backgroundMusic.volume}
            loop
          />
        </Sequence>
      )}
    </AbsoluteFill>
  );
};

const introStyle: React.CSSProperties = {
  justifyContent: "center",
  alignItems: "center",
  textAlign: "center",
  display: "flex",
  zIndex: 20,
  padding: 72,
  background:
    "radial-gradient(circle at 28% 18%, rgba(56,189,248,0.32), transparent 34%), linear-gradient(150deg, #06111f 0%, #0b1020 48%, #020617 100%)",
};

const eyebrowStyle: React.CSSProperties = {
  color: "#7dd3fc",
  fontSize: 30,
  fontWeight: 900,
  letterSpacing: "0.22em",
  marginBottom: 44,
};

const heroCardStyle: React.CSSProperties = {
  width: "92%",
  color: "white",
  textTransform: "uppercase",
  fontSize: 94,
  lineHeight: "102px",
  padding: "42px 36px",
  borderRadius: 42,
  border: "5px solid rgba(125, 211, 252, 0.72)",
  boxShadow: "0 30px 90px rgba(0,0,0,0.42)",
  background: "rgba(15,23,42,0.72)",
};

const introSubStyle: React.CSSProperties = {
  color: "rgba(255,255,255,0.74)",
  fontSize: 34,
  fontWeight: 800,
  marginTop: 44,
};

const SceneOverlay: React.FC<{ index: number; title: string; caption: string }> = ({
  index,
  title,
  caption,
}) => {
  const labels = ["Problem", "Proof", "Next step"];
  return (
    <AbsoluteFill style={{ zIndex: 5, pointerEvents: "none" }}>
      <div style={topBarStyle}>
        <span style={dotStyle} />
        <span>{labels[index] ?? "Proof"}</span>
      </div>
      <div style={productCardStyle}>
        <div style={miniHeaderStyle}>
          <span>{title}</span>
          <span style={livePillStyle}>LIVE DEMO</span>
        </div>
        <div style={mockWindowStyle}>
          <div style={mockRowStyle} />
          <div style={{ ...mockRowStyle, width: "64%" }} />
          <div style={answerBubbleStyle}>{caption || "Show the product doing the work."}</div>
        </div>
      </div>
    </AbsoluteFill>
  );
};

const topBarStyle: React.CSSProperties = {
  position: "absolute",
  top: 82,
  left: 72,
  right: 72,
  display: "flex",
  alignItems: "center",
  gap: 18,
  color: "white",
  fontSize: 30,
  fontWeight: 900,
  letterSpacing: "0.08em",
  textTransform: "uppercase",
};

const dotStyle: React.CSSProperties = {
  width: 18,
  height: 18,
  borderRadius: 99,
  background: "#22d3ee",
  boxShadow: "0 0 34px #22d3ee",
};

const productCardStyle: React.CSSProperties = {
  position: "absolute",
  top: 268,
  left: 82,
  right: 82,
  borderRadius: 46,
  background: "rgba(2,6,23,0.82)",
  border: "3px solid rgba(255,255,255,0.16)",
  boxShadow: "0 36px 90px rgba(0,0,0,0.44)",
  padding: 34,
};

const miniHeaderStyle: React.CSSProperties = {
  display: "flex",
  justifyContent: "space-between",
  alignItems: "center",
  color: "white",
  fontSize: 27,
  fontWeight: 900,
  marginBottom: 28,
};

const livePillStyle: React.CSSProperties = {
  color: "#052e16",
  background: "#86efac",
  borderRadius: 999,
  padding: "10px 16px",
  fontSize: 20,
};

const mockWindowStyle: React.CSSProperties = {
  minHeight: 480,
  borderRadius: 34,
  background: "linear-gradient(180deg, rgba(15,23,42,0.92), rgba(30,41,59,0.92))",
  border: "2px solid rgba(255,255,255,0.12)",
  padding: 34,
};

const mockRowStyle: React.CSSProperties = {
  width: "82%",
  height: 46,
  borderRadius: 999,
  background: "rgba(255,255,255,0.14)",
  marginBottom: 24,
};

const answerBubbleStyle: React.CSSProperties = {
  marginTop: 46,
  marginLeft: "auto",
  width: "78%",
  minHeight: 170,
  borderRadius: 32,
  background: "linear-gradient(135deg, #22d3ee, #a78bfa)",
  color: "#020617",
  fontSize: 34,
  lineHeight: "42px",
  fontWeight: 950,
  padding: 30,
};
