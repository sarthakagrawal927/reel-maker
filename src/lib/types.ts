import { z } from "zod";

export interface AudioTimestamps {
  characters: string[];
  characterStartTimesSeconds: number[];
  characterEndTimesSeconds: number[];
}

const BackgroundTransitionTypeSchema = z.union([
  z.literal("fade"),
  z.literal("blur"),
  z.literal("none"),
]);

const TimelineElementSchema = z.object({
  startMs: z.number(),
  endMs: z.number(),
});

const ElementAnimationSchema = TimelineElementSchema.extend({
  type: z.literal("scale"),
  from: z.number(),
  to: z.number(),
});

const BackgroundElementSchema = TimelineElementSchema.extend({
  imageUrl: z.string(),
  enterTransition: BackgroundTransitionTypeSchema.optional(),
  exitTransition: BackgroundTransitionTypeSchema.optional(),
  animations: z.array(ElementAnimationSchema).optional(),
});

const CaptionTokenSchema = z.object({
  text: z.string(),
  startMs: z.number(),
  endMs: z.number(),
});

const TextElementSchema = TimelineElementSchema.extend({
  captions: z.array(CaptionTokenSchema),
  position: z.union([
    z.literal("top"),
    z.literal("bottom"),
    z.literal("center"),
  ]),
});

const AudioElementSchema = TimelineElementSchema.extend({
  audioUrl: z.string(),
});

export const VideoStyleSchema = z.object({
  highlightColor: z.string().default("#FFE500"),
  captionMaxFontSize: z.number().default(120),
  combineMs: z.number().default(1200),
  captionPosition: z
    .union([z.literal("top"), z.literal("bottom"), z.literal("center")])
    .default("bottom"),
  strokeWidth: z.number().default(15),
  strokeColor: z.string().default("black"),
});

const BackgroundMusicSchema = z.object({
  url: z.string(),
  volume: z.number().min(0).max(1),
});

const TimelineSchema = z.object({
  shortTitle: z.string(),
  elements: z.array(BackgroundElementSchema),
  text: z.array(TextElementSchema),
  audio: z.array(AudioElementSchema),
  style: VideoStyleSchema.optional(),
  backgroundMusic: BackgroundMusicSchema.optional(),
});

export type BackgroundTransitionType = z.infer<
  typeof BackgroundTransitionTypeSchema
>;

export type TimelineElement = z.infer<typeof TimelineElementSchema>;
export type ElementAnimation = z.infer<typeof ElementAnimationSchema>;
export type BackgroundElement = z.infer<typeof BackgroundElementSchema>;
export type CaptionToken = z.infer<typeof CaptionTokenSchema>;
export type TextElement = z.infer<typeof TextElementSchema>;
export type AudioElement = z.infer<typeof AudioElementSchema>;
export type VideoStyle = z.infer<typeof VideoStyleSchema>;
export type BackgroundMusic = z.infer<typeof BackgroundMusicSchema>;
export type Timeline = z.infer<typeof TimelineSchema>;

export {
  AudioElementSchema,
  BackgroundElementSchema,
  BackgroundMusicSchema,
  BackgroundTransitionTypeSchema,
  CaptionTokenSchema,
  ElementAnimationSchema,
  TextElementSchema,
  TimelineElementSchema,
  TimelineSchema,
};

export const StoryScript = z.object({
  text: z.string(),
});

export const StoryWithImages = z.object({
  result: z.array(
    z.object({
      text: z.string(),
      imageDescription: z.string(),
    }),
  ),
});

export const VoiceDescriptorSchema = z.object({
  id: z.string(),
  name: z.string(),
});

export type VoiceDescriptor = z.infer<typeof VoiceDescriptorSchema>;

export interface StoryMetadataWithDetails {
  shortTitle: string;
  content: ContentItemWithDetails[];
}

export interface ContentItemWithDetails {
  text: string;
  imageDescription: string;
  uid: string;
  audioTimestamps: AudioTimestamps;
}
