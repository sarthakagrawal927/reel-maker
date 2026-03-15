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
    <AbsoluteFill style={{ backgroundColor: "black" }}>
      <Sequence durationInFrames={INTRO_DURATION}>
        <AbsoluteFill
          style={{
            justifyContent: "center",
            alignItems: "center",
            textAlign: "center",
            display: "flex",
            zIndex: 10,
          }}
        >
          <div
            style={{
              fontSize: 120,
              lineHeight: "122px",
              width: "87%",
              color: "black",
              fontFamily,
              textTransform: "uppercase",
              backgroundColor: "yellow",
              paddingTop: 20,
              paddingBottom: 20,
              border: "10px solid black",
            }}
          >
            {timeline.shortTitle}
          </div>
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
