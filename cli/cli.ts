#!/usr/bin/env node

import z from "zod";
import yargs from "yargs";
import { hideBin } from "yargs/helpers";
import prompts from "prompts";
import ora from "ora";
import chalk from "chalk";
import * as dotenv from "dotenv";
import * as fs from "fs";
import * as path from "path";
import { execSync } from "child_process";
import {
  AiProvider,
  ImageProvider,
  ImageQuality,
  TtsProvider,
  VideoStyle,
  createGatewayModelWithFallback,
  generateAiImage,
  generateLipSync,
  generateVideoI2V,
  generateVideoT2V,
  generateVoice,
  getGenerateImageDescriptionPrompt,
  getGenerateStoryPrompt,
} from "./service";
import {
  StoryMetadataWithDetails,
  StoryScript,
  StoryWithImages,
  Timeline,
} from "../src/lib/types";
import { v4 as uuidv4 } from "uuid";
import { createTimeLineFromStoryWithDetails, VideoConfig } from "./timeline";

dotenv.config({ quiet: true });

// Available Kokoro voices
export const KOKORO_VOICES = [
  // American Female
  "af_heart", "af_alloy", "af_aoede", "af_bella", "af_jessica",
  "af_nicole", "af_nova", "af_river", "af_sarah", "af_sky",
  // American Male
  "am_adam", "am_echo", "am_eric", "am_fenrir", "am_liam",
  "am_michael", "am_onyx", "am_orion", "am_santa",
  // British Female
  "bf_alice", "bf_emma", "bf_isabella", "bf_lily",
  // British Male
  "bm_daniel", "bm_fable", "bm_george", "bm_lewis",
] as const;

interface GenerateOptions {
  apiKey?: string;
  falKey?: string;
  title?: string;
  topic?: string;
  provider?: AiProvider;
  imageProvider?: "hf" | "stablehorde" | "fal" | "modal";
  imageQuality?: ImageQuality;
  videoStyle?: VideoStyle;
  // TTS
  voice?: string;
  // Style
  captionColor?: string;
  captionSize?: number;
  captionPosition?: "top" | "bottom" | "center";
  strokeWidth?: number;
  strokeColor?: string;
  combineMs?: number;
  // Music
  music?: string;
  musicVolume?: number;
  // Talking head
  avatar?: string;
  // Render
  render?: boolean;
}

class ContentFS {
  title: string;
  slug: string;

  constructor(title: string) {
    this.title = title;
    this.slug = this.getSlug();
  }

  saveDescriptor(descriptor: StoryMetadataWithDetails) {
    const dirPath = this.getDir();
    const filePath = path.join(dirPath, "descriptor.json");
    fs.writeFileSync(filePath, JSON.stringify(descriptor, null, 2));
  }

  loadDescriptor(): StoryMetadataWithDetails | null {
    const filePath = path.join(this.getDir(), "descriptor.json");
    if (!fs.existsSync(filePath)) return null;
    try {
      return JSON.parse(fs.readFileSync(filePath, "utf-8")) as StoryMetadataWithDetails;
    } catch {
      return null;
    }
  }

  saveTimeline(timeline: Timeline) {
    const dirPath = this.getDir();
    const filePath = path.join(dirPath, "timeline.json");
    fs.writeFileSync(filePath, JSON.stringify(timeline, null, 2));
  }

  copyMusic(srcPath: string) {
    const destDir = this.getDir();
    const destPath = path.join(destDir, "bg-music.mp3");
    fs.copyFileSync(srcPath, destPath);
  }

  getDir(dir?: string): string {
    const segments = ["public", "content", this.slug];
    if (dir) segments.push(dir);
    const p = path.join(process.cwd(), ...segments);
    fs.mkdirSync(p, { recursive: true });
    return p;
  }

  getImagePath(uid: string): string {
    return path.join(this.getDir("images"), `${uid}.png`);
  }

  getAudioPath(uid: string): string {
    return path.join(this.getDir("audio"), `${uid}.mp3`);
  }

  getVideoPath(uid: string): string {
    return path.join(this.getDir("videos"), `${uid}.mp4`);
  }

  getSlug(): string {
    return this.title
      .toLowerCase()
      .replace(/[^a-z0-9]+/g, "-")
      .replace(/^-+|-+$/g, "");
  }
}

export async function generateStory(options: GenerateOptions) {
  try {
    let apiKey = options.apiKey ?? process.env.FREE_GATEWAY_API_KEY;

    if (!apiKey) {
      const response = await prompts({
        type: "password",
        name: "apiKey",
        message: "Enter your Free Gateway API key:",
        validate: (value) => value.length > 0 || "API key is required",
      });
      if (!response.apiKey) {
        console.log(chalk.red("API key is required. Exiting..."));
        process.exit(1);
      }
      apiKey = response.apiKey;
    }

    // Resolve image provider: HF > Stable Horde > Modal > fal.ai > prompt
    let imageProvider: ImageProvider;
    const hfToken = process.env.HF_TOKEN;
    const modalUrl = process.env.MODAL_IMAGE_GEN_URL;
    const falKey = options.falKey ?? process.env.FAL_KEY;
    const stableHordeKey = process.env.STABLE_HORDE_API_KEY;

    if (options.imageProvider === "hf" || (!options.imageProvider && hfToken)) {
      imageProvider = { type: "hf", token: hfToken! };
      console.log(chalk.blue("Images: HuggingFace (FLUX.1-schnell, free)"));
    } else if (options.imageProvider === "stablehorde" || (!options.imageProvider && stableHordeKey)) {
      imageProvider = { type: "stablehorde", apiKey: stableHordeKey! };
      console.log(chalk.blue("Images: Stable Horde (FLUX.1-schnell, free, slow queue)"));
    } else if (options.imageProvider === "modal" || (!options.imageProvider && modalUrl)) {
      const url = modalUrl!;
      const quality = options.imageQuality ?? "fast";
      imageProvider = { type: "modal", url, quality };
      const modelLabel = quality === "quality" ? "FLUX.1-dev" : "FLUX.1-schnell";
      console.log(chalk.blue(`Images: Modal (${modelLabel})`));
    } else if (options.imageProvider === "fal" || (!options.imageProvider && falKey)) {
      imageProvider = { type: "fal", falKey: falKey! };
      console.log(chalk.blue("Images: fal.ai (Flux Pro)"));
    } else {
      const { choice } = await prompts({
        type: "select",
        name: "choice",
        message: "Choose image provider:",
        choices: [
          { title: "Stable Horde — FLUX.1-schnell (free, needs key)", value: "stablehorde" },
          { title: "HuggingFace — FLUX.1-schnell (free, needs HF_TOKEN)", value: "hf" },
          { title: "Modal — FLUX.1-schnell (~$0.002/image)", value: "modal" },
          { title: "fal.ai — Flux Pro ($0.03/image)", value: "fal" },
        ],
      });
      if (choice === "stablehorde") {
        const { key } = await prompts({
          type: "password",
          name: "key",
          message: "Enter your Stable Horde API key:",
          validate: (v: string) => v.length > 0 || "Key is required",
        });
        imageProvider = { type: "stablehorde", apiKey: key };
      } else if (choice === "hf") {
        const { token } = await prompts({
          type: "password",
          name: "token",
          message: "Enter your HuggingFace token:",
          validate: (v: string) => v.startsWith("hf_") || "Must start with hf_",
        });
        imageProvider = { type: "hf", token };
      } else if (choice === "modal") {
        const { url } = await prompts({
          type: "text",
          name: "url",
          message: "Enter your Modal endpoint URL:",
          validate: (v: string) => v.startsWith("https://") || "Must be a valid HTTPS URL",
        });
        imageProvider = { type: "modal", url };
      } else {
        const { key } = await prompts({
          type: "password",
          name: "key",
          message: "Enter your fal.ai API key:",
          validate: (v: string) => v.length > 0 || "Key is required",
        });
        imageProvider = { type: "fal", falKey: key };
      }
    }

    // Resolve TTS: Edge TTS (Microsoft, free, no key) > Modal (Kokoro-82M)
    let ttsProvider: TtsProvider;
    ttsProvider = { type: "edge" };
    console.log(chalk.blue("TTS: Microsoft Edge (free, no key)"));

    let { title, topic } = options;

    if (!title || !topic) {
      const response = await prompts([
        {
          type: "text",
          name: "title",
          message: "Title of the story:",
          initial: title,
          validate: (value) => value.length > 0 || "Title is required",
        },
        {
          type: "text",
          name: "topic",
          message: "Topic of the story:",
          initial: topic,
          validate: (value) => value.length > 0 || "Topic is required",
        },
      ]);

      if (!response.title || !response.topic) {
        console.log(chalk.red("Title and topic are required. Exiting..."));
        process.exit(1);
      }

      title = response.title;
      topic = response.topic;
    }

    const voice = options.voice ?? "en-GB-SoniaNeural";
    const videoStyle: VideoStyle = options.videoStyle ?? "images";
    const t2vUrl = process.env.MODAL_T2V_URL;
    const i2vUrl = process.env.MODAL_I2V_URL;
    const lipsyncUrl = process.env.MODAL_LIPSYNC_URL;

    if (videoStyle === "t2v" && !t2vUrl) {
      console.log(chalk.red("MODAL_T2V_URL not set in .env"));
      process.exit(1);
    }
    if (videoStyle === "i2v" && !i2vUrl) {
      console.log(chalk.red("MODAL_I2V_URL not set in .env"));
      process.exit(1);
    }
    if (videoStyle === "talking-head" && !lipsyncUrl) {
      console.log(chalk.red("MODAL_LIPSYNC_URL not set in .env (deploy modal/latentsync.py first)"));
      process.exit(1);
    }

    console.log(chalk.blue(`\n📖 Creating story: "${title}"`));
    console.log(chalk.blue(`📝 Topic: ${topic} | Voice: ${voice} | Video: ${videoStyle}\n`));

    const contentFs = new ContentFS(title!);
    const runCompletion = <T>(prompt: string, schema: z.ZodType<T>) =>
      createGatewayModelWithFallback(prompt, schema, apiKey!);

    // ── Cache: reuse existing descriptor if already generated ───────────────
    let storyWithDetails: StoryMetadataWithDetails;
    const cachedDescriptor = contentFs.loadDescriptor();
    if (cachedDescriptor) {
      console.log(chalk.yellow("♻️  Using cached story & image descriptions"));
      storyWithDetails = cachedDescriptor;
    } else {
      storyWithDetails = { shortTitle: title!, content: [] };

      const storySpinner = ora("Generating story...").start();
      const storyRes = await runCompletion(
        getGenerateStoryPrompt(title!, topic!),
        StoryScript,
      );
      storySpinner.succeed(chalk.green("Story generated!"));

      const descriptionsSpinner = ora("Generating image descriptions...").start();
      const storyWithImagesRes = await runCompletion(
        getGenerateImageDescriptionPrompt(storyRes.text),
        StoryWithImages,
      );
      descriptionsSpinner.succeed(chalk.green("Image descriptions generated!"));

      for (const item of storyWithImagesRes.result) {
        storyWithDetails.content.push({
          text: item.text,
          imageDescription: item.imageDescription,
          uid: uuidv4(),
          audioTimestamps: {
            characters: [],
            characterStartTimesSeconds: [],
            characterEndTimesSeconds: [],
          },
        });
      }

      contentFs.saveDescriptor(storyWithDetails);
    }

    // Copy background music if provided
    if (options.music) {
      const musicPath = path.resolve(options.music);
      if (!fs.existsSync(musicPath)) {
        console.log(chalk.red(`Music file not found: ${musicPath}`));
        process.exit(1);
      }
      contentFs.copyMusic(musicPath);
    }

    const imagesSpinner = ora("Generating images and voice...").start();
    // talking-head with --avatar: 2 steps (voice + lipsync); without avatar: 3 (image + voice + lipsync)
    // t2v: 2 steps (voice + video); i2v: 3 (image + voice + video); images: 2 (image + voice)
    const stepsPerItem =
      videoStyle === "t2v" ? 2
      : videoStyle === "i2v" ? 3
      : videoStyle === "talking-head" ? (options.avatar ? 2 : 3)
      : 2;
    const total = storyWithDetails.content.length * stepsPerItem;
    let step = 0;

    for (let i = 0; i < storyWithDetails.content.length; i++) {
      const storyItem = storyWithDetails.content[i];

      // Step 1: generate image (skip for t2v; talking-head skips if --avatar supplied)
      if (videoStyle !== "t2v" && !(videoStyle === "talking-head" && options.avatar)) {
        step++;
        const imgPath = contentFs.getImagePath(storyItem.uid);
        if (fs.existsSync(imgPath)) {
          imagesSpinner.text = `[${step}/${total}] Image cached, skipping...`;
        } else {
          imagesSpinner.text = `[${step}/${total}] Generating image...`;
          await generateAiImage({
            prompt: storyItem.imageDescription,
            path: imgPath,
            provider: imageProvider,
            onRetry: (attempt) => {
              imagesSpinner.text = `[${step}/${total}] Generating image (retry ${attempt})...`;
            },
          });
        }
      }

      // Step 2: voice (always)
      step++;
      const audioPath = contentFs.getAudioPath(storyItem.uid);
      let timings = storyItem.audioTimestamps;
      const hasAudio = fs.existsSync(audioPath) &&
        timings.characters.length > 0;
      if (hasAudio) {
        imagesSpinner.text = `[${step}/${total}] Voice cached, skipping...`;
      } else {
        imagesSpinner.text = `[${step}/${total}] Generating voice...`;
        timings = await generateVoice(storyItem.text, ttsProvider, audioPath, voice);
        storyItem.audioTimestamps = timings;
      }

      // Step 3: video (i2v, t2v, or talking-head)
      if (videoStyle === "i2v" || videoStyle === "t2v" || videoStyle === "talking-head") {
        step++;
        const videoPath = contentFs.getVideoPath(storyItem.uid);
        if (fs.existsSync(videoPath)) {
          imagesSpinner.text = `[${step}/${total}] Video cached, skipping...`;
        } else {
          const durationMs = Math.ceil(
            timings.characterEndTimesSeconds[timings.characterEndTimesSeconds.length - 1] * 1000,
          );
          imagesSpinner.text = `[${step}/${total}] Generating video (${videoStyle})...`;
          if (videoStyle === "i2v") {
            await generateVideoI2V(
              contentFs.getImagePath(storyItem.uid),
              storyItem.imageDescription,
              i2vUrl!,
              videoPath,
              durationMs,
            );
          } else if (videoStyle === "t2v") {
            await generateVideoT2V(
              storyItem.imageDescription,
              t2vUrl!,
              videoPath,
              durationMs,
            );
          } else {
            const avatarPath = options.avatar
              ? path.resolve(options.avatar)
              : contentFs.getImagePath(storyItem.uid);
            await generateLipSync(
              avatarPath,
              audioPath,
              lipsyncUrl!,
              videoPath,
            );
          }
        }
      }
    }

    contentFs.saveDescriptor(storyWithDetails);
    imagesSpinner.succeed(chalk.green("Images and voice generated!"));

    const finalSpinner = ora("Building timeline...").start();

    const videoConfig: VideoConfig = {
      voice,
      videoStyle,
      style: {
        ...(options.captionColor ? { highlightColor: options.captionColor } : {}),
        ...(options.captionSize ? { captionMaxFontSize: options.captionSize } : {}),
        ...(options.captionPosition ? { captionPosition: options.captionPosition } : {}),
        ...(options.strokeWidth !== undefined ? { strokeWidth: options.strokeWidth } : {}),
        ...(options.strokeColor ? { strokeColor: options.strokeColor } : {}),
        ...(options.combineMs ? { combineMs: options.combineMs } : {}),
      },
      ...(options.music
        ? {
            backgroundMusic: {
              localPath: options.music,
              volume: options.musicVolume ?? 0.15,
            },
          }
        : {}),
    };

    const timeline = createTimeLineFromStoryWithDetails(storyWithDetails, videoConfig);
    contentFs.saveTimeline(timeline);
    finalSpinner.succeed(chalk.green("Timeline built!"));

    console.log(chalk.green.bold("\n✨ Story generation complete!\n"));

    if (options.render) {
      console.log(chalk.blue("🎬 Rendering video..."));
      execSync(`bunx remotion render ${contentFs.slug}`, { stdio: "inherit" });
      console.log(chalk.green.bold("🎥 Render complete!\n"));
    } else {
      console.log("Preview: " + chalk.blue("bun run dev"));
      console.log("Render:  " + chalk.blue(`bunx remotion render ${contentFs.slug}`));
    }

    return { slug: contentFs.slug };
  } catch (error) {
    console.error(chalk.red("\n❌ Error:"), error);
    process.exit(1);
  }
}

const styleOptions = (y: ReturnType<typeof import("yargs")>) =>
  y
    .option("image-quality", {
      type: "string",
      choices: ["fast", "quality"] as const,
      description: "fast = FLUX.1-schnell (4 steps, $0.002); quality = FLUX.1-dev (28 steps, $0.01)",
    })
    .option("video-style", {
      type: "string",
      choices: ["images", "i2v", "t2v", "talking-head"] as const,
      description: "images = static (default); i2v = animate each image; t2v = generate video from prompt; talking-head = lip-synced avatar",
    })
    .option("avatar", {
      type: "string",
      description: "Path to portrait image for talking-head mode (optional; generates one if omitted)",
    })
    .option("voice", {
      type: "string",
      description: `TTS voice ID (default: en-GB-SoniaNeural). Edge TTS voices recommended (e.g. en-US-JennyNeural, en-GB-LibbyNeural). Kokoro options: ${KOKORO_VOICES.join(", ")}`,
    })
    .option("caption-color", {
      type: "string",
      description: "Word highlight color (default: #FFE500)",
    })
    .option("caption-size", {
      type: "number",
      description: "Max caption font size in px (default: 120)",
    })
    .option("caption-position", {
      type: "string",
      choices: ["top", "bottom", "center"] as const,
      description: "Caption position (default: bottom)",
    })
    .option("stroke-width", {
      type: "number",
      description: "Caption stroke width in px (default: 15)",
    })
    .option("stroke-color", {
      type: "string",
      description: "Caption stroke color (default: black)",
    })
    .option("combine-ms", {
      type: "number",
      description: "Combine words within this ms window (default: 1200)",
    })
    .option("music", {
      type: "string",
      description: "Path to background music MP3 file",
    })
    .option("music-volume", {
      type: "number",
      description: "Background music volume 0–1 (default: 0.15)",
    })
    .option("render", {
      type: "boolean",
      description: "Auto-render video after generation",
    });

const commonOptions = (y: ReturnType<typeof import("yargs")>) =>
  styleOptions(y)
    .option("api-key", {
      alias: "k",
      type: "string",
      description: "AI provider API key",
    })
    .option("fal-key", {
      alias: "f",
      type: "string",
      description: "fal.ai API key for image generation",
    })
    .option("image-provider", {
      type: "string",
      choices: ["hf", "stablehorde", "modal", "fal"] as const,
      description: "Image generation provider (default: hf → stablehorde → modal → fal)",
    })
    .option("title", {
      alias: "t",
      type: "string",
      description: "Title of the story",
    })
    .option("topic", {
      alias: "p",
      type: "string",
      description: "Topic of the story (e.g. Interesting Facts, History)",
    })
    .option("provider", {
      type: "string",
      description: "AI provider for story generation (only 'gateway' supported)",
    });

yargs(hideBin(process.argv))
  .command(
    "generate",
    "Generate a story reel from title and topic",
    commonOptions,
    async (argv) => {
      await generateStory({
        apiKey: argv["api-key"],
        falKey: argv["fal-key"],
        title: argv.title,
        topic: argv.topic,
        provider: argv.provider as AiProvider | undefined,
        imageProvider: argv["image-provider"] as "hf" | "stablehorde" | "fal" | "modal" | undefined,
        imageQuality: argv["image-quality"] as ImageQuality | undefined,
        videoStyle: argv["video-style"] as VideoStyle | undefined,
        voice: argv.voice,
        avatar: argv.avatar,
        captionColor: argv["caption-color"],
        captionSize: argv["caption-size"],
        captionPosition: argv["caption-position"] as "top" | "bottom" | "center" | undefined,
        strokeWidth: argv["stroke-width"],
        strokeColor: argv["stroke-color"],
        combineMs: argv["combine-ms"],
        music: argv.music,
        musicVolume: argv["music-volume"],
        render: argv.render,
      });
    },
  )
  .command(
    "$0",
    "Generate a story reel (default)",
    commonOptions,
    async (argv) => {
      await generateStory({
        apiKey: argv["api-key"],
        falKey: argv["fal-key"],
        title: argv.title,
        topic: argv.topic,
        provider: argv.provider as AiProvider | undefined,
        imageProvider: argv["image-provider"] as "hf" | "stablehorde" | "fal" | "modal" | undefined,
        imageQuality: argv["image-quality"] as ImageQuality | undefined,
        videoStyle: argv["video-style"] as VideoStyle | undefined,
        voice: argv.voice,
        avatar: argv.avatar,
        captionColor: argv["caption-color"],
        captionSize: argv["caption-size"],
        captionPosition: argv["caption-position"] as "top" | "bottom" | "center" | undefined,
        strokeWidth: argv["stroke-width"],
        strokeColor: argv["stroke-color"],
        combineMs: argv["combine-ms"],
        music: argv.music,
        musicVolume: argv["music-volume"],
        render: argv.render,
      });
    },
  )
  .demandCommand(0, 1)
  .help()
  .alias("help", "h")
  .version()
  .alias("version", "v")
  .strict()
  .parse();
