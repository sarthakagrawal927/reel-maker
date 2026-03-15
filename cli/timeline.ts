import type {
  BackgroundElement,
  CaptionToken,
  ElementAnimation,
  StoryMetadataWithDetails,
  TextElement,
  Timeline,
  VideoStyle,
} from "../src/lib/types";
import { VideoStyleSchema } from "../src/lib/types";
import type { VideoStyle as VideoStyleMode } from "./service";

export interface VideoConfig {
  voice?: string;
  videoStyle?: VideoStyleMode;
  style?: Partial<VideoStyle>;
  backgroundMusic?: { localPath: string; volume: number };
  render?: boolean;
}

const characterTimestampsToWordCaptions = (
  characters: string[],
  startTimes: number[],
  endTimes: number[],
  offsetMs: number,
): CaptionToken[] => {
  const captions: CaptionToken[] = [];
  let currentWord = "";
  let wordStartMs = 0;
  let isFirst = true;

  for (let i = 0; i < characters.length; i++) {
    const char = characters[i];
    const charStartMs = startTimes[i] * 1000 + offsetMs;

    if (char === " ") {
      if (currentWord) {
        const lastEndMs = endTimes[i - 1] * 1000 + offsetMs;
        captions.push({
          text: isFirst ? currentWord : ` ${currentWord}`,
          startMs: wordStartMs,
          endMs: lastEndMs,
        });
        isFirst = false;
        currentWord = "";
      }
    } else {
      if (!currentWord) wordStartMs = charStartMs;
      currentWord += char;
    }
  }

  if (currentWord) {
    const lastEndMs = endTimes[endTimes.length - 1] * 1000 + offsetMs;
    captions.push({
      text: isFirst ? currentWord : ` ${currentWord}`,
      startMs: wordStartMs,
      endMs: lastEndMs,
    });
  }

  return captions;
};

export const createTimeLineFromStoryWithDetails = (
  storyWithDetails: StoryMetadataWithDetails,
  config?: VideoConfig,
): Timeline => {
  const style = config?.style
    ? VideoStyleSchema.parse({ ...config.style })
    : undefined;

  const timeline: Timeline = {
    elements: [],
    text: [],
    audio: [],
    shortTitle: storyWithDetails.shortTitle,
    ...(style ? { style } : {}),
    ...(config?.backgroundMusic
      ? {
          backgroundMusic: {
            url: "bg-music.mp3",
            volume: config.backgroundMusic.volume,
          },
        }
      : {}),
  };

  let durationMs = 0;
  let zoomIn = true;

  for (let i = 0; i < storyWithDetails.content.length; i++) {
    const content = storyWithDetails.content[i];

    const lenMs = Math.ceil(
      content.audioTimestamps.characterEndTimesSeconds[
        content.audioTimestamps.characterEndTimesSeconds.length - 1
      ] * 1000,
    );

    const useVideo = config?.videoStyle === "i2v" || config?.videoStyle === "t2v";
    const bgElem: BackgroundElement = {
      startMs: durationMs,
      endMs: durationMs + lenMs,
      ...(config?.videoStyle !== "t2v" ? { imageUrl: content.uid } : {}),
      ...(useVideo ? { videoUrl: content.uid } : {}),
      enterTransition: "blur",
      exitTransition: "blur",
      animations: useVideo ? [] : getBgAnimations(lenMs, zoomIn),
    };

    timeline.elements.push(bgElem);

    timeline.audio.push({
      startMs: durationMs,
      endMs: durationMs + lenMs,
      audioUrl: content.uid,
    });

    const captions = characterTimestampsToWordCaptions(
      content.audioTimestamps.characters,
      content.audioTimestamps.characterStartTimesSeconds,
      content.audioTimestamps.characterEndTimesSeconds,
      durationMs,
    );

    const textElem: TextElement = {
      startMs: durationMs,
      endMs: durationMs + lenMs,
      captions,
      position: style?.captionPosition ?? "bottom",
    };

    timeline.text.push(textElem);

    durationMs += lenMs;
    zoomIn = !zoomIn;
  }

  return timeline;
};

export const getBgAnimations = (durationMs: number, zoomIn: boolean) => {
  const animations: ElementAnimation[] = [];

  animations.push({
    type: "scale",
    from: zoomIn ? 1.5 : 1,
    to: zoomIn ? 1 : 1.5,
    startMs: 0,
    endMs: durationMs,
  });

  return animations;
};
