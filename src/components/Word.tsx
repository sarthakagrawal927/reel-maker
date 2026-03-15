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
import type { VideoStyle } from "../lib/types";

const { fontFamily } = loadFont();

export const Word: React.FC<{ page: TikTokPage; style: VideoStyle }> = ({
  page,
  style,
}) => {
  const frame = useCurrentFrame();
  const { fps, width, height } = useVideoConfig();
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

  const fontSize = Math.min(style.captionMaxFontSize, fittedText.fontSize);

  const positionStyle: React.CSSProperties =
    style.captionPosition === "top"
      ? { top: 200, bottom: undefined, height: 150 }
      : style.captionPosition === "center"
        ? { top: height / 2 - 75, bottom: undefined, height: 150 }
        : { top: undefined, bottom: 350, height: 150 };

  return (
    <AbsoluteFill
      style={{
        justifyContent: "center",
        alignItems: "center",
        ...positionStyle,
      }}
    >
      <div
        style={{
          fontSize,
          WebkitTextStroke: `${style.strokeWidth}px ${style.strokeColor}`,
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
                color: active ? style.highlightColor : "white",
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
