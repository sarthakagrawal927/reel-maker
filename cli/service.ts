import z from "zod";
import * as fs from "fs";
import { IMAGE_HEIGHT, IMAGE_WIDTH } from "../src/lib/constants";
import type { AudioTimestamps } from "../src/lib/types";
import { generateObject, jsonSchema } from "ai";
import { createOpenAI } from "@ai-sdk/openai";
import type { LanguageModel } from "ai";
import { fal } from "@fal-ai/client";

export type AiProvider = "gateway";

export const createGatewayModelWithFallback = async <T>(
  prompt: string,
  schema: z.ZodType<T>,
  apiKey: string,
): Promise<T> => {
  const baseURL = process.env.FREE_GATEWAY_URL!;
  // Try up to 3 times — gateway picks a different model each attempt
  let lastError: Error | null = null;
  for (let i = 0; i < 3; i++) {
    try {
      const model = createOpenAI({ apiKey, baseURL })("");
      return await structuredCompletion(prompt, schema, model);
    } catch (err) {
      lastError = err instanceof Error ? err : new Error(String(err));
    }
  }
  throw lastError!;
};

export const structuredCompletion = async <T>(
  prompt: string,
  schema: z.ZodType<T>,
  model: LanguageModel,
): Promise<T> => {
  const rawSchema = z.toJSONSchema(schema) as Record<string, unknown>;
  const { object } = await generateObject({
    model,
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    schema: jsonSchema<T>(rawSchema as any),
    mode: "json",
    prompt,
    maxTokens: 4096,
  });
  return object;
};

export type ImageProvider =
  | { type: "hf"; token: string }
  | { type: "stablehorde"; apiKey: string }
  | { type: "modal"; url: string; quality?: ImageQuality }
  | { type: "fal"; falKey: string };

export type TtsProvider =
  | { type: "edge" }
  | { type: "modal"; url: string };


export type ImageQuality = "fast" | "quality"; // schnell vs dev
export type VideoStyle = "images" | "i2v" | "t2v" | "talking-head";

// Modal web endpoints return HTTP 303 for long-running tasks.
// We follow the Location header and poll with GET until we get a 200.
const modalFetchWithPolling = async <T>(
  url: string,
  body: object,
  timeoutMs = 300_000,
): Promise<T> => {
  const deadline = Date.now() + timeoutMs;

  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
    redirect: "manual",
  });

  if (res.status === 200) {
    return res.json() as Promise<T>;
  }

  if (res.status !== 303) {
    throw new Error(`Modal error ${res.status}: ${await res.text()}`);
  }

  // Poll the redirect location
  const pollUrl = res.headers.get("location");
  if (!pollUrl) throw new Error("Modal 303 missing Location header");

  while (Date.now() < deadline) {
    await new Promise((r) => setTimeout(r, 2000));
    const poll = await fetch(pollUrl);
    if (poll.status === 200) {
      return poll.json() as Promise<T>;
    }
    if (poll.status !== 202) {
      throw new Error(`Modal poll error ${poll.status}: ${await poll.text()}`);
    }
  }

  throw new Error("Modal request timed out");
};

export const generateAiImage = async ({
  prompt,
  path,
  provider,
  onRetry,
}: {
  prompt: string;
  path: string;
  provider: ImageProvider;
  onRetry: (attempt: number) => void;
}): Promise<void> => {
  const maxRetries = 3;
  let attempt = 0;
  let lastError: Error | null = null;

  while (attempt < maxRetries) {
    try {
      if (provider.type === "stablehorde") {
        // Stable Horde — community GPU grid, free with API key
        // Free tier limit: ≤576×576 area, dims must be multiples of 64. Max portrait: 448×576
        const submitRes = await fetch("https://stablehorde.net/api/v2/generate/async", {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            apikey: provider.apiKey,
          },
          body: JSON.stringify({
            prompt,
            params: {
              width: 448,
              height: 576,
              steps: 4,
              sampler_name: "k_euler",
              cfg_scale: 1,
            },
            models: ["Flux.1-Schnell"],
            r2: false,
          }),
        });
        if (!submitRes.ok) throw new Error(`Stable Horde submit error ${submitRes.status}: ${await submitRes.text()}`);
        const { id } = await submitRes.json() as { id: string };

        // Poll until done
        const deadline = Date.now() + 300_000;
        while (Date.now() < deadline) {
          await new Promise((r) => setTimeout(r, 4000));
          const checkRes = await fetch(`https://stablehorde.net/api/v2/generate/check/${id}`, {
            headers: { apikey: provider.apiKey },
          });
          const check = await checkRes.json() as { done: boolean; faulted?: boolean };
          if (check.faulted) throw new Error("Stable Horde job faulted");
          if (!check.done) continue;

          const statusRes = await fetch(`https://stablehorde.net/api/v2/generate/status/${id}`, {
            headers: { apikey: provider.apiKey },
          });
          const status = await statusRes.json() as { generations: { img: string }[] };
          const imgUrl = status.generations[0].img;
          const imgRes = await fetch(imgUrl);
          fs.writeFileSync(path, Buffer.from(await imgRes.arrayBuffer()));
          break;
        }
      } else if (provider.type === "hf") {
        // HuggingFace Inference API — FLUX.1-schnell, free tier
        const res = await fetch(
          "https://router.huggingface.co/hf-inference/models/black-forest-labs/FLUX.1-schnell",
          {
            method: "POST",
            headers: {
              Authorization: `Bearer ${provider.token}`,
              "Content-Type": "application/json",
              "x-wait-for-model": "true",
            },
            body: JSON.stringify({
              inputs: prompt,
              parameters: { width: IMAGE_WIDTH, height: IMAGE_HEIGHT, num_inference_steps: 4 },
            }),
          },
        );
        if (!res.ok) throw new Error(`HF API error ${res.status}: ${await res.text()}`);
        fs.writeFileSync(path, Buffer.from(await res.arrayBuffer()));
      } else if (provider.type === "modal") {
        const data = await modalFetchWithPolling<{ image_b64: string }>(provider.url, {
          prompt,
          width: IMAGE_WIDTH,
          height: IMAGE_HEIGHT,
          model: provider.quality === "quality" ? "dev" : "schnell",
          steps: provider.quality === "quality" ? 28 : 4,
        });
        fs.writeFileSync(path, Buffer.from(data.image_b64, "base64"));
      } else {
        fal.config({ credentials: provider.falKey });
        const result = await fal.subscribe("fal-ai/flux-pro/v1.1", {
          input: {
            prompt,
            image_size: { width: IMAGE_WIDTH, height: IMAGE_HEIGHT },
            num_images: 1,
          },
        });
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        const imageUrl = (result.data as any).images[0].url as string;
        const response = await fetch(imageUrl);
        fs.writeFileSync(path, Buffer.from(await response.arrayBuffer()));
      }
      return;
    } catch (err) {
      lastError = err instanceof Error ? err : new Error(String(err));
      attempt++;
      if (attempt < maxRetries) {
        const isRateLimit = lastError.message.includes("429");
        await new Promise((resolve) => setTimeout(resolve, isRateLimit ? 15_000 : 2_000));
        onRetry(attempt);
      }
    }
  }

  throw lastError!;
};

export const getGenerateStoryPrompt = (title: string, topic: string) =>
  `Write a short story with title [${title}] (its topic is [${topic}]).
   You must follow best practices for great storytelling.
   The script must be 8-10 sentences long.
   Story events can be from anywhere in the world, but text must be translated into English language.
   Result without any formatting and title, as one continuous text.
   Skip new lines.`;

export const getGenerateImageDescriptionPrompt = (storyText: string) =>
  `You are given story text.
  Generate (in English) exactly 5 very detailed image descriptions for this story.
  Return their description as json array with story sentences matched to images.
  Story sentences must be in the same order as in the story and their content must be preserved.
  Each image must match 1-2 sentence from the story.
  Images must show story content in a way that is visually appealing and engaging, not just characters.
  Give output in json format:

  {
    "result": [
      {
        "text": "....",
        "imageDescription": "..."
      }
    ]
  }

  <story>
  ${storyText}
  </story>`;


// Default Edge TTS voice — can be overridden with any en-US-*Neural name
const EDGE_DEFAULT_VOICE = "en-US-AriaNeural";

export const generateVoice = async (
  text: string,
  provider: TtsProvider,
  audioPath: string,
  voice = "en-GB-SoniaNeural",
): Promise<AudioTimestamps> => {
  if (provider.type === "edge") {
    const { MsEdgeTTS, OUTPUT_FORMAT } = await import("msedge-tts");
    const tts = new MsEdgeTTS();

    // Use Edge voice name if passed (en-US-AriaNeural etc), else default
    const edgeVoice = voice.includes("-") ? voice : EDGE_DEFAULT_VOICE;
    await tts.setMetadata(edgeVoice, OUTPUT_FORMAT.AUDIO_24KHZ_48KBITRATE_MONO_MP3, {
      wordBoundaryEnabled: true,
    });

    const { audioStream, metadataStream } = tts.toStream(text);

    // Collect audio and metadata in parallel
    const audioChunks: Buffer[] = [];
    const metaChunks: Buffer[] = [];
    audioStream.on("data", (d: Buffer) => audioChunks.push(d));
    metadataStream?.on("data", (d: Buffer) => metaChunks.push(d));
    await new Promise<void>((resolve, reject) => {
      audioStream.on("close", resolve);
      audioStream.on("error", reject);
    });

    fs.writeFileSync(audioPath, Buffer.concat(audioChunks));

    // Parse word-boundary metadata: Offset/Duration are in 100-nanosecond units → seconds
    const characters: string[] = [];
    const characterStartTimesSeconds: number[] = [];
    const characterEndTimesSeconds: number[] = [];

    type WBEntry = { Type: string; Data: { Offset: number; Duration: number; text: { Text: string } } };
    const wordBoundaries: WBEntry[] = metaChunks.flatMap((buf) => {
      try {
        const parsed = JSON.parse(buf.toString()) as { Metadata: WBEntry[] };
        return (parsed.Metadata ?? []).filter((e) => e.Type === "WordBoundary");
      } catch { return []; }
    });

    for (let wi = 0; wi < wordBoundaries.length; wi++) {
      const wb = wordBoundaries[wi];
      const word = wb.Data.text.Text;
      if (!word) continue;
      const ws = wb.Data.Offset / 10_000_000;
      const we = ws + wb.Data.Duration / 10_000_000;
      const dur = (we - ws) / Math.max(word.length, 1);
      for (let i = 0; i < word.length; i++) {
        characters.push(word[i]);
        characterStartTimesSeconds.push(ws + i * dur);
        characterEndTimesSeconds.push(ws + (i + 1) * dur);
      }
      if (wi < wordBoundaries.length - 1) {
        characters.push(" ");
        characterStartTimesSeconds.push(we);
        characterEndTimesSeconds.push(we);
      }
    }

    return { characters, characterStartTimesSeconds, characterEndTimesSeconds };
  }

  // Modal path
  const data = await modalFetchWithPolling<{
    audio_b64: string;
    characters: string[];
    characterStartTimesSeconds: number[];
    characterEndTimesSeconds: number[];
  }>(provider.url, { text, voice });

  fs.writeFileSync(audioPath, Buffer.from(data.audio_b64, "base64"));

  return {
    characters: data.characters,
    characterStartTimesSeconds: data.characterStartTimesSeconds,
    characterEndTimesSeconds: data.characterEndTimesSeconds,
  };
};

// ─── num_frames helper: Wan requires 4k+1 frames ─────────────────────────────
const toWanFrames = (durationMs: number, fps = 16, max = 81): number => {
  const raw = Math.round((durationMs / 1000) * fps);
  const clamped = Math.min(max, Math.max(17, raw));
  return Math.floor((clamped - 1) / 4) * 4 + 1; // round down to nearest 4k+1
};

// ─── Text-to-video ─────────────────────────────────────────────────────────────
export const generateVideoT2V = async (
  prompt: string,
  t2vUrl: string,
  outPath: string,
  durationMs: number,
): Promise<void> => {
  const data = await modalFetchWithPolling<{ video_b64: string }>(
    t2vUrl,
    { prompt, width: 480, height: 832, num_frames: toWanFrames(durationMs) },
    600_000,
  );
  fs.mkdirSync(require("path").dirname(outPath), { recursive: true });
  fs.writeFileSync(outPath, Buffer.from(data.video_b64, "base64"));
};

// ─── Talking head (LatentSync) ────────────────────────────────────────────────
export const generateLipSync = async (
  imagePath: string,
  audioPath: string,
  lipsyncUrl: string,
  outPath: string,
): Promise<void> => {
  const image_b64 = fs.readFileSync(imagePath).toString("base64");
  const audio_b64 = fs.readFileSync(audioPath).toString("base64");
  const data = await modalFetchWithPolling<{ video_b64: string }>(
    lipsyncUrl,
    { image_b64, audio_b64 },
    300_000,
  );
  fs.mkdirSync(require("path").dirname(outPath), { recursive: true });
  fs.writeFileSync(outPath, Buffer.from(data.video_b64, "base64"));
};

// ─── Image-to-video ────────────────────────────────────────────────────────────
export const generateVideoI2V = async (
  imagePath: string,
  prompt: string,
  i2vUrl: string,
  outPath: string,
  durationMs: number,
): Promise<void> => {
  const image_b64 = fs.readFileSync(imagePath).toString("base64");
  const data = await modalFetchWithPolling<{ video_b64: string }>(
    i2vUrl,
    {
      image_b64,
      prompt,
      width: 480,
      height: 832,
      num_frames: toWanFrames(durationMs),
    },
    600_000,
  );
  fs.mkdirSync(require("path").dirname(outPath), { recursive: true });
  fs.writeFileSync(outPath, Buffer.from(data.video_b64, "base64"));
};
