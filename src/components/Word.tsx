import type { TikTokPage } from "@remotion/captions";
import { makeTransform, scale, translateY } from "@remotion/animation-utils";
import { fitText } from "@remotion/layout-utils";
import { loadFont } from "@remotion/google-fonts/BreeSerif";
import type React from "react";
import {
  AbsoluteFill,
  interpolate,
  spring,
  useCurrentFrame,
  useVideoConfig,
} from "remotion";

const HIGHLIGHT_COLOR = "#FFE500";

export const Word: React.FC<{ page: TikTokPage }> = ({ page }) => {
  const frame = useCurrentFrame();
  const { fps, width } = useVideoConfig();
  const { fontFamily } = loadFont();
  const timeInMs = (frame / fps) * 1000;

  const enter = spring({
    frame,
    fps,
    config: { damping: 200 },
    durationInFrames: 5,
  });

  const fittedText = fitText({
    fontFamily,
    text: page.text,
    withinWidth: width * 0.85,
    textTransform: "uppercase",
  });

  const fontSize = Math.min(120, fittedText.fontSize);

  return (
    <AbsoluteFill
      style={{
        justifyContent: "center",
        alignItems: "center",
        top: undefined,
        bottom: 350,
        height: 150,
      }}
    >
      <div
        style={{
          fontSize,
          WebkitTextStroke: "15px black",
          paintOrder: "stroke fill",
          transform: makeTransform([
            scale(interpolate(enter, [0, 1], [0.8, 1])),
            translateY(interpolate(enter, [0, 1], [50, 0])),
          ]),
          fontFamily,
          textTransform: "uppercase",
          textAlign: "center",
        }}
      >
        {page.tokens.map((token) => {
          const active = token.fromMs <= timeInMs && token.toMs > timeInMs;
          return (
            <span
              key={token.fromMs}
              style={{
                display: "inline",
                whiteSpace: "pre",
                color: active ? HIGHLIGHT_COLOR : "white",
              }}
            >
              {token.text}
            </span>
          );
        })}
      </div>
    </AbsoluteFill>
  );
};
