#!/usr/bin/env node
/**
 * Reddit → Reel pipeline.
 *
 * Fetches top posts from funny/story subreddits, paraphrases them into
 * narration scripts, and generates reels via the existing reel-maker pipeline.
 *
 * Usage:
 *   bun run reddit                              # Interactive (defaults)
 *   bun run reddit --count 3 --render           # Top 3 posts, auto-render
 *   bun run reddit --subreddits "tifu,AmItheAsshole" --count 5
 *   bun run reddit --dry-run                    # Preview posts without generating
 */

import yargs from "yargs";
import { hideBin } from "yargs/helpers";
import chalk from "chalk";
import ora from "ora";
import * as fs from "fs";
import * as path from "path";
import * as dotenv from "dotenv";
import { v4 as uuidv4 } from "uuid";
import {
  createGatewayModelWithFallback,
  generateAiImage,
  generateVoice,
  generateVideoI2V,
  generateVideoT2V,
  generateLipSync,
  getGenerateImageDescriptionPrompt,
  type ImageProvider,
  type TtsProvider,
  type VideoStyle,
  type ImageQuality,
} from "./service";
import {
  StoryMetadataWithDetails,
  StoryWithImages,
} from "../src/lib/types";
import { createTimeLineFromStoryWithDetails, type VideoConfig } from "./timeline";

dotenv.config({ quiet: true });

// ── Config ────────────────────────────────────────────────────────────────────

const DEFAULT_SUBREDDITS = [
  "tifu",
  "AmItheAsshole",
  "MaliciousCompliance",
  "pettyrevenge",
  "ProRevenge",
  "entitledparents",
  "TalesFromRetail",
  "TalesFromTechSupport",
  "BestofRedditorUpdates",
];

const MIN_SCORE = 500;
const MIN_TEXT_LENGTH = 200;
const MAX_TEXT_LENGTH = 3000;
const HISTORY_FILE = path.join(import.meta.dir, "..", "reddit-history.json");
const USER_AGENT = "ReelBot/1.0";

// ── Types ─────────────────────────────────────────────────────────────────────

interface RedditPostData {
  id: string;
  subreddit: string;
  title: string;
  selftext: string;
  score: number;
  is_self: boolean;
  over_18: boolean;
  num_comments: number;
  permalink: string;
}

interface HistoryEntry {
  redditId: string;
  subreddit: string;
  title: string;
  slug: string;
  processedAt: string;
}

interface History {
  processed: HistoryEntry[];
}

// ── Reddit Fetcher ────────────────────────────────────────────────────────────

async function fetchTopPosts(
  subreddit: string,
  limit = 25,
  sort: "hot" | "top" = "hot",
): Promise<RedditPostData[]> {
  const url =
    sort === "top"
      ? `https://www.reddit.com/r/${subreddit}/top.json?t=day&limit=${limit}`
      : `https://www.reddit.com/r/${subreddit}/hot.json?limit=${limit}`;

  const res = await fetch(url, {
    headers: { "User-Agent": USER_AGENT },
  });

  if (!res.ok) {
    throw new Error(`Reddit fetch failed for r/${subreddit}: ${res.status}`);
  }

  const json = (await res.json()) as {
    data: { children: { data: RedditPostData }[] };
  };

  return json.data.children.map((c) => c.data);
}

// ── Filtering ─────────────────────────────────────────────────────────────────

function filterPosts(
  posts: RedditPostData[],
  processedIds: Set<string>,
): RedditPostData[] {
  return posts.filter((p) => {
    if (!p.is_self) return false; // text posts only
    if (p.over_18) return false;
    if (p.score < MIN_SCORE) return false;
    if (p.selftext.length < MIN_TEXT_LENGTH) return false;
    if (p.selftext.length > MAX_TEXT_LENGTH) return false;
    if (p.selftext === "[removed]" || p.selftext === "[deleted]") return false;
    if (processedIds.has(p.id)) return false;
    return true;
  });
}

// ── History ───────────────────────────────────────────────────────────────────

function loadHistory(): History {
  if (!fs.existsSync(HISTORY_FILE)) return { processed: [] };
  try {
    return JSON.parse(fs.readFileSync(HISTORY_FILE, "utf-8")) as History;
  } catch {
    return { processed: [] };
  }
}

function saveHistory(history: History) {
  fs.writeFileSync(HISTORY_FILE, JSON.stringify(history, null, 2));
}

// ── Script Paraphraser ────────────────────────────────────────────────────────

const getParaphrasePrompt = (title: string, body: string) =>
  `You are a voiceover scriptwriter for short-form video (30-90 seconds).

Rewrite this internet story into a punchy narration script. Rules:
- Paraphrase — do NOT copy verbatim. Change names, details, phrasing.
- Keep the core story and humor/drama intact.
- Write in first person if the original is first person, otherwise third person.
- Conversational tone — like telling a friend a wild story.
- 6-10 sentences. No filler, no intro like "so this happened", just dive in.
- Output as plain text, no formatting, no title, no newlines between sentences.

Title: ${title}

Story:
${body}`;

// ── Content FS (duplicated from cli.ts to avoid modifying core) ───────────────

class ContentFS {
  title: string;
  slug: string;

  constructor(title: string) {
    this.title = title;
    this.slug = this.getSlug();
  }

  saveDescriptor(descriptor: StoryMetadataWithDetails) {
    const filePath = path.join(this.getDir(), "descriptor.json");
    fs.writeFileSync(filePath, JSON.stringify(descriptor, null, 2));
  }

  loadDescriptor(): StoryMetadataWithDetails | null {
    const filePath = path.join(this.getDir(), "descriptor.json");
    if (!fs.existsSync(filePath)) return null;
    try {
      return JSON.parse(
        fs.readFileSync(filePath, "utf-8"),
      ) as StoryMetadataWithDetails;
    } catch {
      return null;
    }
  }

  saveTimeline(timeline: unknown) {
    const filePath = path.join(this.getDir(), "timeline.json");
    fs.writeFileSync(filePath, JSON.stringify(timeline, null, 2));
  }

  copyMusic(srcPath: string) {
    fs.copyFileSync(srcPath, path.join(this.getDir(), "bg-music.mp3"));
  }

  getDir(dir?: string): string {
    const segments = ["public", "content", this.slug];
    if (dir) segments.push(dir);
    const p = path.join(process.cwd(), ...segments);
    fs.mkdirSync(p, { recursive: true });
    return p;
  }

  getImagePath(uid: string) {
    return path.join(this.getDir("images"), `${uid}.png`);
  }
  getAudioPath(uid: string) {
    return path.join(this.getDir("audio"), `${uid}.mp3`);
  }
  getVideoPath(uid: string) {
    return path.join(this.getDir("videos"), `${uid}.mp4`);
  }

  private getSlug(): string {
    return this.title
      .toLowerCase()
      .replace(/[^a-z0-9]+/g, "-")
      .replace(/^-+|-+$/g, "")
      .slice(0, 60);
  }
}

// ── Generate a single reel from a Reddit post ─────────────────────────────────

interface ReelOptions {
  voice: string;
  videoStyle: VideoStyle;
  imageProvider: ImageProvider;
  ttsProvider: TtsProvider;
  render: boolean;
  music?: string;
  musicVolume?: number;
  captionColor?: string;
  captionSize?: number;
  captionPosition?: "top" | "bottom" | "center";
  strokeWidth?: number;
  strokeColor?: string;
}

async function generateRedditReel(
  post: RedditPostData,
  apiKey: string,
  options: ReelOptions,
): Promise<string> {
  const runCompletion = <T>(prompt: string, schema: import("zod").ZodType<T>) =>
    createGatewayModelWithFallback(prompt, schema, apiKey);

  // Step 1: Paraphrase the post into a narration script
  const paraphraseSpinner = ora("Paraphrasing story...").start();
  let script: string;
  try {
    const models = ["gemini-2.5-flash", "groq-llama-70b", "workers-ai-llama-3.3-70b"];
    const { createOpenAI } = await import("@ai-sdk/openai");
    const { generateText } = await import("ai");
    let lastErr: Error | null = null;
    script = "";
    for (const modelId of models) {
      try {
        const model = createOpenAI({
          apiKey,
          baseURL: process.env.FREE_GATEWAY_URL!,
        })(modelId);
        const { text } = await generateText({
          model,
          prompt: getParaphrasePrompt(post.title, post.selftext),
        });
        script = text.trim();
        break;
      } catch (err) {
        lastErr = err instanceof Error ? err : new Error(String(err));
      }
    }
    if (!script) throw lastErr!;
    paraphraseSpinner.succeed(chalk.green("Story paraphrased!"));
  } catch (err) {
    paraphraseSpinner.fail(chalk.red("Failed to paraphrase"));
    throw err;
  }

  // Step 2: Generate a short reel title (not the Reddit title)
  const titleSlug = post.title
    .replace(/^(TIFU|AITA|TIL)\s*/i, "")
    .slice(0, 50)
    .trim();
  const reelTitle = `reddit-${post.subreddit}-${post.id}`;

  const contentFs = new ContentFS(reelTitle);

  // Step 3: Check cache
  let storyWithDetails: StoryMetadataWithDetails;
  const cached = contentFs.loadDescriptor();
  if (cached) {
    console.log(chalk.yellow("  Using cached descriptor"));
    storyWithDetails = cached;
  } else {
    // Generate image descriptions from paraphrased script
    const imgSpinner = ora("Generating image descriptions...").start();
    const storyWithImages = await runCompletion(
      getGenerateImageDescriptionPrompt(script),
      StoryWithImages,
    );
    imgSpinner.succeed(chalk.green("Image descriptions ready!"));

    storyWithDetails = {
      shortTitle: titleSlug,
      content: storyWithImages.result.map((item) => ({
        text: item.text,
        imageDescription: item.imageDescription,
        uid: uuidv4(),
        audioTimestamps: {
          characters: [],
          characterStartTimesSeconds: [],
          characterEndTimesSeconds: [],
        },
      })),
    };
    contentFs.saveDescriptor(storyWithDetails);
  }

  // Step 4: Generate assets (images, voice, optional video)
  if (options.music) {
    const musicPath = path.resolve(options.music);
    if (fs.existsSync(musicPath)) contentFs.copyMusic(musicPath);
  }

  const assetsSpinner = ora("Generating assets...").start();
  const stepsPerItem =
    options.videoStyle === "t2v"
      ? 2
      : options.videoStyle === "i2v"
        ? 3
        : 2;
  const total = storyWithDetails.content.length * stepsPerItem;
  let step = 0;

  for (const item of storyWithDetails.content) {
    // Image
    if (options.videoStyle !== "t2v") {
      step++;
      const imgPath = contentFs.getImagePath(item.uid);
      if (fs.existsSync(imgPath)) {
        assetsSpinner.text = `[${step}/${total}] Image cached`;
      } else {
        assetsSpinner.text = `[${step}/${total}] Generating image...`;
        await generateAiImage({
          prompt: item.imageDescription,
          path: imgPath,
          provider: options.imageProvider,
          onRetry: (attempt, providerType) => {
            assetsSpinner.text = `[${step}/${total}] ${attempt === 0 ? `Falling back to ${providerType}` : `Retry ${attempt} via ${providerType}`}`;
          },
        });
      }
    }

    // Voice
    step++;
    const audioPath = contentFs.getAudioPath(item.uid);
    const hasAudio =
      fs.existsSync(audioPath) && item.audioTimestamps.characters.length > 0;
    if (hasAudio) {
      assetsSpinner.text = `[${step}/${total}] Voice cached`;
    } else {
      assetsSpinner.text = `[${step}/${total}] Generating voice...`;
      item.audioTimestamps = await generateVoice(
        item.text,
        options.ttsProvider,
        audioPath,
        options.voice,
      );
    }

    // Video (i2v/t2v)
    if (options.videoStyle === "i2v" || options.videoStyle === "t2v") {
      step++;
      const videoPath = contentFs.getVideoPath(item.uid);
      if (fs.existsSync(videoPath)) {
        assetsSpinner.text = `[${step}/${total}] Video cached`;
      } else {
        const durationMs = Math.ceil(
          item.audioTimestamps.characterEndTimesSeconds[
            item.audioTimestamps.characterEndTimesSeconds.length - 1
          ] * 1000,
        );
        assetsSpinner.text = `[${step}/${total}] Generating video (${options.videoStyle})...`;
        if (options.videoStyle === "i2v") {
          await generateVideoI2V(
            contentFs.getImagePath(item.uid),
            item.imageDescription,
            process.env.MODAL_I2V_URL!,
            videoPath,
            durationMs,
          );
        } else {
          await generateVideoT2V(
            item.imageDescription,
            process.env.MODAL_T2V_URL!,
            videoPath,
            durationMs,
          );
        }
      }
    }
  }

  contentFs.saveDescriptor(storyWithDetails);
  assetsSpinner.succeed(chalk.green("Assets generated!"));

  // Step 5: Build timeline
  const timelineSpinner = ora("Building timeline...").start();
  const videoConfig: VideoConfig = {
    voice: options.voice,
    videoStyle: options.videoStyle,
    style: {
      ...(options.captionColor ? { highlightColor: options.captionColor } : {}),
      ...(options.captionSize
        ? { captionMaxFontSize: options.captionSize }
        : {}),
      ...(options.captionPosition
        ? { captionPosition: options.captionPosition }
        : {}),
      ...(options.strokeWidth !== undefined
        ? { strokeWidth: options.strokeWidth }
        : {}),
      ...(options.strokeColor ? { strokeColor: options.strokeColor } : {}),
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

  const timeline = createTimeLineFromStoryWithDetails(
    storyWithDetails,
    videoConfig,
  );
  contentFs.saveTimeline(timeline);
  timelineSpinner.succeed(chalk.green("Timeline built!"));

  // Step 6: Render
  if (options.render) {
    console.log(chalk.blue("  Rendering video..."));
    const { execSync } = await import("child_process");
    execSync(`bunx remotion render ${contentFs.slug}`, { stdio: "inherit" });
    console.log(chalk.green.bold("  Render complete!"));
  }

  return contentFs.slug;
}

// ── Main ──────────────────────────────────────────────────────────────────────

async function main(argv: {
  subreddits?: string;
  count: number;
  sort: "hot" | "top";
  dryRun: boolean;
  render: boolean;
  voice?: string;
  videoStyle?: string;
  imageProvider?: string;
  imageQuality?: string;
  captionColor?: string;
  captionSize?: number;
  captionPosition?: string;
  strokeWidth?: number;
  strokeColor?: string;
  music?: string;
  musicVolume?: number;
}) {
  const apiKey = process.env.FREE_GATEWAY_API_KEY;
  if (!apiKey && !argv.dryRun) {
    console.error(chalk.red("FREE_GATEWAY_API_KEY not set in .env"));
    process.exit(1);
  }

  const subreddits = argv.subreddits
    ? argv.subreddits.split(",").map((s) => s.trim())
    : DEFAULT_SUBREDDITS;

  const history = loadHistory();
  const processedIds = new Set(history.processed.map((h) => h.redditId));

  // Fetch posts from all subreddits
  const fetchSpinner = ora(
    `Fetching ${argv.sort} posts from ${subreddits.length} subreddits...`,
  ).start();

  let allPosts: RedditPostData[] = [];
  for (const sub of subreddits) {
    try {
      const posts = await fetchTopPosts(sub, 25, argv.sort);
      const filtered = filterPosts(posts, processedIds);
      allPosts.push(...filtered);
    } catch (err) {
      // Skip failed subreddits silently
    }
    // Small delay to avoid rate limiting
    await new Promise((r) => setTimeout(r, 500));
  }

  // Sort by score descending and take top N
  allPosts.sort((a, b) => b.score - a.score);
  allPosts = allPosts.slice(0, argv.count);

  fetchSpinner.succeed(
    chalk.green(
      `Found ${allPosts.length} eligible posts across ${subreddits.length} subreddits`,
    ),
  );

  if (allPosts.length === 0) {
    console.log(chalk.yellow("No new posts matching criteria. Try --sort top or lower score threshold."));
    return;
  }

  // Preview
  console.log(chalk.bold("\nSelected posts:"));
  for (const p of allPosts) {
    console.log(
      `  ${chalk.cyan(`r/${p.subreddit}`)} ${chalk.white(p.title.slice(0, 60))} ${chalk.gray(`(score: ${p.score}, ${p.selftext.length} chars)`)}`,
    );
  }

  if (argv.dryRun) {
    console.log(chalk.yellow("\n--dry-run: skipping generation"));
    return;
  }

  // Resolve providers
  const hfToken = process.env.HF_TOKEN;
  const modalUrl = process.env.MODAL_IMAGE_GEN_URL;
  const falKey = process.env.FAL_KEY;
  const stableHordeKey = process.env.STABLE_HORDE_API_KEY;

  let imageProvider: ImageProvider;
  if (argv.imageProvider === "hf" || (!argv.imageProvider && hfToken)) {
    imageProvider = { type: "hf", token: hfToken! };
  } else if (argv.imageProvider === "stablehorde" || (!argv.imageProvider && stableHordeKey)) {
    imageProvider = { type: "stablehorde", apiKey: stableHordeKey! };
  } else if (argv.imageProvider === "modal" || (!argv.imageProvider && modalUrl)) {
    imageProvider = { type: "modal", url: modalUrl!, quality: (argv.imageQuality as "fast" | "quality") ?? "fast" };
  } else if (argv.imageProvider === "fal" || (!argv.imageProvider && falKey)) {
    imageProvider = { type: "fal", falKey: falKey! };
  } else {
    console.error(chalk.red("No image provider configured. Set HF_TOKEN, STABLE_HORDE_API_KEY, MODAL_IMAGE_GEN_URL, or FAL_KEY in .env"));
    process.exit(1);
  }

  const ttsProvider: TtsProvider = { type: "edge" };
  const voice = argv.voice ?? "en-GB-SoniaNeural";
  const videoStyle = (argv.videoStyle as VideoStyle) ?? "images";

  // Generate reels
  console.log(chalk.bold(`\nGenerating ${allPosts.length} reel(s)...\n`));

  const results: { title: string; status: "ok" | "error"; slug?: string; error?: string }[] = [];

  for (let i = 0; i < allPosts.length; i++) {
    const post = allPosts[i];
    console.log(
      chalk.bold.blue(
        `\n[${i + 1}/${allPosts.length}] r/${post.subreddit}: ${post.title.slice(0, 50)}`,
      ),
    );

    try {
      const slug = await generateRedditReel(post, apiKey, {
        voice,
        videoStyle,
        imageProvider,
        ttsProvider,
        render: argv.render,
        music: argv.music,
        musicVolume: argv.musicVolume,
        captionColor: argv.captionColor,
        captionSize: argv.captionSize,
        captionPosition: argv.captionPosition as "top" | "bottom" | "center" | undefined,
        strokeWidth: argv.strokeWidth,
        strokeColor: argv.strokeColor,
      });

      // Save to history
      history.processed.push({
        redditId: post.id,
        subreddit: post.subreddit,
        title: post.title,
        slug,
        processedAt: new Date().toISOString(),
      });
      saveHistory(history);

      results.push({ title: post.title, status: "ok", slug });
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      results.push({ title: post.title, status: "error", error: msg });
      console.error(chalk.red(`  Failed: ${msg}`));
    }
  }

  // Summary
  console.log(chalk.bold("\nSummary:"));
  for (const r of results) {
    const icon = r.status === "ok" ? chalk.green("done") : chalk.red("fail");
    console.log(
      `  ${icon} ${r.title.slice(0, 50)}${r.slug ? chalk.gray(` → ${r.slug}`) : ""}${r.error ? chalk.red(` — ${r.error}`) : ""}`,
    );
  }

  const ok = results.filter((r) => r.status === "ok").length;
  console.log(
    chalk.bold(`\n${ok}/${results.length} reels generated successfully.`),
  );
  if (ok > 0 && !argv.render) {
    console.log(chalk.blue("Run with --render to auto-render, or:"));
    console.log(chalk.blue("  bun run dev    # Preview in Remotion Studio"));
    console.log(
      chalk.blue(
        `  bunx remotion render <slug>    # Render a specific reel`,
      ),
    );
  }
}

// ── CLI ───────────────────────────────────────────────────────────────────────

yargs(hideBin(process.argv))
  .option("subreddits", {
    alias: "s",
    type: "string",
    description: `Comma-separated subreddits (default: ${DEFAULT_SUBREDDITS.join(",")})`,
  })
  .option("count", {
    alias: "n",
    type: "number",
    default: 5,
    description: "Number of reels to generate",
  })
  .option("sort", {
    type: "string",
    choices: ["hot", "top"] as const,
    default: "hot" as const,
    description: "Reddit sort order (hot = trending, top = highest score today)",
  })
  .option("dry-run", {
    type: "boolean",
    default: false,
    description: "Preview selected posts without generating",
  })
  .option("render", {
    type: "boolean",
    default: false,
    description: "Auto-render videos after generation",
  })
  .option("voice", {
    type: "string",
    description: "TTS voice (default: en-GB-SoniaNeural)",
  })
  .option("video-style", {
    type: "string",
    choices: ["images", "i2v", "t2v"] as const,
    description: "Video style (default: images)",
  })
  .option("image-provider", {
    type: "string",
    choices: ["hf", "stablehorde", "modal", "fal"] as const,
    description: "Image provider",
  })
  .option("image-quality", {
    type: "string",
    choices: ["fast", "quality"] as const,
  })
  .option("caption-color", { type: "string" })
  .option("caption-size", { type: "number" })
  .option("caption-position", {
    type: "string",
    choices: ["top", "bottom", "center"] as const,
  })
  .option("stroke-width", { type: "number" })
  .option("stroke-color", { type: "string" })
  .option("music", { type: "string", description: "Path to background music MP3" })
  .option("music-volume", { type: "number" })
  .help()
  .parseAsync()
  .then((argv) =>
    main({
      subreddits: argv.subreddits as string | undefined,
      count: argv.count as number,
      sort: argv.sort as "hot" | "top",
      dryRun: argv["dry-run"] as boolean,
      render: argv.render as boolean,
      voice: argv.voice as string | undefined,
      videoStyle: argv["video-style"] as string | undefined,
      imageProvider: argv["image-provider"] as string | undefined,
      imageQuality: argv["image-quality"] as string | undefined,
      captionColor: argv["caption-color"] as string | undefined,
      captionSize: argv["caption-size"] as number | undefined,
      captionPosition: argv["caption-position"] as string | undefined,
      strokeWidth: argv["stroke-width"] as number | undefined,
      strokeColor: argv["stroke-color"] as string | undefined,
      music: argv.music as string | undefined,
      musicVolume: argv["music-volume"] as number | undefined,
    }),
  );
