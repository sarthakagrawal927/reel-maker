import z from "zod";
import * as fs from "fs";
import { IMAGE_HEIGHT, IMAGE_WIDTH } from "../src/lib/constants";
import type { AudioTimestamps } from "../src/lib/types";
import { generateObject, jsonSchema } from "ai";
import { createOpenAI } from "@ai-sdk/openai";
import { createAnthropic } from "@ai-sdk/anthropic";
import { createGoogleGenerativeAI } from "@ai-sdk/google";
import type { LanguageModel } from "ai";
import { fal } from "@fal-ai/client";

export type AiProvider = "openai" | "anthropic" | "google";

export const createModel = (
  provider: AiProvider,
  apiKey: string,
  modelId?: string,
): LanguageModel => {
  if (provider === "anthropic") {
    return createAnthropic({ apiKey })(modelId ?? "claude-sonnet-4-5-20251001");
  }
  if (provider === "google") {
    return createGoogleGenerativeAI({ apiKey })(modelId ?? "gemini-2.5-flash");
  }
  return createOpenAI({ apiKey })(modelId ?? "gpt-4.1");
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
    prompt,
  });
  return object;
};

export type ImageProvider =
  | { type: "fal"; falKey: string }
  | { type: "modal"; url: string };

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
      if (provider.type === "modal") {
        // Modal web endpoints return 303 for long-running tasks; poll until done
        const data = await modalFetchWithPolling<{ image_b64: string }>(provider.url, {
          prompt,
          width: IMAGE_WIDTH,
          height: IMAGE_HEIGHT,
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
        await new Promise((resolve) => setTimeout(resolve, 1000));
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
  Generate (in English) 5-8 very detailed image descriptions for this story.
  Return their description as json array with story sentences matched to images.
  Story sentences must be in the same order as in the story and their content must be preserved.
  Each image must match 1-2 sentence from the story.
  Images must show story content in a way that is visually appealing and engaging, not just characters.
  Give output in json format:

  [
    {
      "text": "....",
      "imageDescription": "..."
    }
  ]

  <story>
  ${storyText}
  </story>`;

export const generateVoice = async (
  text: string,
  ttsUrl: string,
  path: string,
  voice = "af_heart",
): Promise<AudioTimestamps> => {
  const data = await modalFetchWithPolling<{
    audio_b64: string;
    characters: string[];
    characterStartTimesSeconds: number[];
    characterEndTimesSeconds: number[];
  }>(ttsUrl, { text, voice });

  fs.writeFileSync(path, Buffer.from(data.audio_b64, "base64"));

  return {
    characters: data.characters,
    characterStartTimesSeconds: data.characterStartTimesSeconds,
    characterEndTimesSeconds: data.characterEndTimesSeconds,
  };
};
