import z from "zod";
import * as fs from "fs";
import { ElevenLabsClient } from "@elevenlabs/elevenlabs-js";
import { CharacterAlignmentResponseModel } from "@elevenlabs/elevenlabs-js/api";
import { IMAGE_HEIGHT, IMAGE_WIDTH } from "../src/lib/constants";
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
    return createGoogleGenerativeAI({ apiKey })(modelId ?? "gemini-2.0-flash");
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
        const res = await fetch(provider.url, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            prompt,
            width: IMAGE_WIDTH,
            height: IMAGE_HEIGHT,
          }),
        });
        if (!res.ok) throw new Error(`Modal error: ${await res.text()}`);
        const data = (await res.json()) as { image_b64: string };
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
  apiKey: string,
  path: string,
): Promise<CharacterAlignmentResponseModel> => {
  const client = new ElevenLabsClient({
    environment: "https://api.elevenlabs.io",
    apiKey,
  });

  const voiceId = "21m00Tcm4TlvDq8ikWAM";

  const data = await client.textToSpeech.convertWithTimestamps(voiceId, {
    text,
  });

  if (!data.alignment || !data.alignment.characterEndTimesSeconds.length) {
    throw new Error("ElevenLabs response missing timestamps");
  }

  fs.writeFileSync(path, Buffer.from(data.audioBase64, "base64"));
  return data.alignment;
};
