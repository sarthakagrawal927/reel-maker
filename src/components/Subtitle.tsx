import type { Caption } from "@remotion/captions";
import { createTikTokStyleCaptions } from "@remotion/captions";
import React from "react";
import { AbsoluteFill, Sequence, useVideoConfig } from "remotion";
import type { CaptionToken } from "../lib/types";
import { Word } from "./Word";

const COMBINE_MS = 1200;

const Subtitle: React.FC<{
  captions: CaptionToken[];
  sceneStartMs: number;
}> = ({ captions, sceneStartMs }) => {
  const { fps } = useVideoConfig();

  const remotionCaptions: Caption[] = captions.map((c) => ({
    text: c.text,
    startMs: c.startMs - sceneStartMs,
    endMs: c.endMs - sceneStartMs,
    timestampMs: c.startMs - sceneStartMs,
    confidence: 1,
  }));

  const { pages } = createTikTokStyleCaptions({
    captions: remotionCaptions,
    combineTokensWithinMilliseconds: COMBINE_MS,
  });

  return (
    <AbsoluteFill>
      {pages.map((page, i) => {
        const nextPage = pages[i + 1] ?? null;
        const startFrame = Math.round((page.startMs / 1000) * fps);
        const endFrame = nextPage
          ? Math.round((nextPage.startMs / 1000) * fps)
          : Math.round(((page.startMs + page.durationMs) / 1000) * fps);
        const durationInFrames = Math.max(1, endFrame - startFrame);

        return (
          <Sequence
            key={i}
            from={startFrame}
            durationInFrames={durationInFrames}
          >
            <Word page={page} />
          </Sequence>
        );
      })}
    </AbsoluteFill>
  );
};

export default Subtitle;
