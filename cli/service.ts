import z from "zod";
import * as fs from "fs";
import { IMAGE_HEIGHT, IMAGE_WIDTH } from "../src/lib/constants";
import type { AudioTimestamps } from "../src/lib/types";
import { generateText } from "ai";
import { createOpenAI } from "@ai-sdk/openai";
import type { LanguageModel } from "ai";
import { fal } from "@fal-ai/client";

export type AiProvider = "gateway";

// High-reasoning models to try in order
const HIGH_REASONING_MODELS = ["gemini-2.5-flash", "groq-llama-70b", "workers-ai-llama-3.3-70b"];

const getGatewayModel = (apiKey: string, modelId: string): LanguageModel =>
  createOpenAI({ apiKey, baseURL: process.env.FREE_GATEWAY_URL! })(modelId);

// Plain text generation (for story scripts — avoids JSON mode issues)
export const generateStoryScript = async (prompt: string, apiKey: string): Promise<string> => {
  let lastError: Error | null = null;
  for (const modelId of HIGH_REASONING_MODELS) {
    try {
      const { text } = await generateText({ model: getGatewayModel(apiKey, modelId), prompt });
      return text.trim();
    } catch (err) {
      lastError = err instanceof Error ? err : new Error(String(err));
    }
  }
  throw lastError!;
};

export const createGatewayModelWithFallback = async <T>(
  prompt: string,
  schema: z.ZodType<T>,
  apiKey: string,
): Promise<T> => {
  let lastError: Error | null = null;
  for (const modelId of HIGH_REASONING_MODELS) {
    try {
      return await structuredCompletion(prompt, schema, getGatewayModel(apiKey, modelId));
    } catch (err) {
      lastError = err instanceof Error ? err : new Error(String(err));
    }
  }
  throw lastError!;
};

const stripJsonCodeBlock = (text: string): string => {
  // Strip ```json ... ``` or ``` ... ``` wrappers
  const match = text.match(/```(?:json)?\s*([\s\S]*?)```/);
  return match ? match[1].trim() : text.trim();
};

export const structuredCompletion = async <T>(
  prompt: string,
  schema: z.ZodType<T>,
  model: LanguageModel,
): Promise<T> => {
  const { text } = await generateText({ model, prompt, maxTokens: 4096 });
  const json = stripJsonCodeBlock(text);
  const parsed = JSON.parse(json);
  return schema.parse(parsed);
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

const NO_TEXT_SUFFIX = ", no text, no words, no letters, no labels, no watermarks, no captions";

const tryGenerateImage = async (prompt: string, path: string, provider: ImageProvider): Promise<void> => {
  prompt = prompt + NO_TEXT_SUFFIX;
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
  onRetry: (attempt: number, providerType: string) => void;
}): Promise<void> => {
  // Build provider fallback chain from the selected provider
  const chain: ImageProvider[] = [provider];
  // Add downstream fallbacks if not already in chain
  const fallbacks: ImageProvider[] = [];
  if (provider.type !== "stablehorde" && process.env.STABLE_HORDE_API_KEY)
    fallbacks.push({ type: "stablehorde", apiKey: process.env.STABLE_HORDE_API_KEY });
  if (provider.type !== "modal" && process.env.MODAL_IMAGE_GEN_URL)
    fallbacks.push({ type: "modal", url: process.env.MODAL_IMAGE_GEN_URL, quality: "fast" });
  if (provider.type !== "fal" && process.env.FAL_KEY)
    fallbacks.push({ type: "fal", falKey: process.env.FAL_KEY });

  const allProviders = [...chain, ...fallbacks.filter(f => !chain.some(c => c.type === f.type))];
  let lastError: Error | null = null;

  for (const p of allProviders) {
    let attempt = 0;
    while (attempt < 3) {
      try {
        await tryGenerateImage(prompt, path, p);
        return;
      } catch (err) {
        lastError = err instanceof Error ? err : new Error(String(err));
        // Non-retryable: quota/auth errors — skip to next provider
        if (lastError.message.includes("402") || lastError.message.includes("401") || lastError.message.includes("403")) break;
        attempt++;
        if (attempt < 3) {
          const isRateLimit = lastError.message.includes("429");
          await new Promise((r) => setTimeout(r, isRateLimit ? 15_000 : 2_000));
          onRetry(attempt, p.type);
        }
      }
    }
    if (allProviders.indexOf(p) < allProviders.length - 1) {
      onRetry(0, allProviders[allProviders.indexOf(p) + 1].type);
    }
  }

  throw lastError!;
};

export const getGenerateStoryPrompt = (title: string, topic: string) =>
  `Write a punchy, engaging lesson script about [${title}] (topic: [${topic}]).
   Style: direct and clear like a Fireship or Kurzgesagt explainer — no fluff, no filler.
   Structure: hook (1 sentence that grabs attention) → what it is → why it matters → how it works → real-world example → key takeaway.
   The script must be 8-10 sentences total.
   Do NOT use storytelling wrappers like "In a bustling city..." or fictional characters.
   Speak directly to the viewer. Use concrete examples, numbers, and analogies where helpful.
   Output as plain text with no formatting, no title, no newlines.`;

export const getGenerateImageDescriptionPrompt = (storyText: string) =>
  `You are given story text.
  Generate (in English) exactly 5 very detailed image descriptions for this lesson script.
  Return their description as json array with script segments matched to images.
  Script segments must be in the same order as in the script and their content must be preserved.
  Each image must match 1-2 sentences from the script.
  Images should be conceptual and visual — diagrams, metaphors, abstract representations of technical concepts. Avoid generic people at computers. Think bold infographic-style visuals.
  IMPORTANT: Do not include any text, words, letters, labels, numbers, or writing in the images. Pure visual only.
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
