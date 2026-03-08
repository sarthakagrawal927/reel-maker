#!/usr/bin/env node

import yargs from "yargs";
import { hideBin } from "yargs/helpers";
import prompts from "prompts";
import ora from "ora";
import chalk from "chalk";
import * as dotenv from "dotenv";
import {
  AiProvider,
  createModel,
  generateAiImage,
  generateVoice,
  getGenerateImageDescriptionPrompt,
  getGenerateStoryPrompt,
  structuredCompletion,
} from "./service";
import {
  ContentItemWithDetails,
  StoryMetadataWithDetails,
  StoryScript,
  StoryWithImages,
  Timeline,
} from "../src/lib/types";
import { v4 as uuidv4 } from "uuid";
import * as fs from "fs";
import * as path from "path";
import { createTimeLineFromStoryWithDetails } from "./timeline";

dotenv.config({ quiet: true });

interface GenerateOptions {
  apiKey?: string;
  falKey?: string;
  elevenlabsApiKey?: string;
  title?: string;
  topic?: string;
  provider?: AiProvider;
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

  saveTimeline(timeline: Timeline) {
    const dirPath = this.getDir();
    const filePath = path.join(dirPath, "timeline.json");
    fs.writeFileSync(filePath, JSON.stringify(timeline, null, 2));
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

  getSlug(): string {
    return this.title
      .toLowerCase()
      .replace(/[^a-z0-9]+/g, "-")
      .replace(/^-+|-+$/g, "");
  }
}

async function generateStory(options: GenerateOptions) {
  try {
    const provider: AiProvider =
      options.provider ??
      (process.env.ANTHROPIC_API_KEY ? "anthropic" : "openai");

    let apiKey =
      options.apiKey ??
      (provider === "anthropic"
        ? process.env.ANTHROPIC_API_KEY
        : process.env.OPENAI_API_KEY);
    let falKey = options.falKey ?? process.env.FAL_KEY;
    let elevenlabsApiKey =
      options.elevenlabsApiKey ?? process.env.ELEVENLABS_API_KEY;

    if (!apiKey) {
      const keyName = provider === "anthropic" ? "Anthropic" : "OpenAI";
      const response = await prompts({
        type: "password",
        name: "apiKey",
        message: `Enter your ${keyName} API key:`,
        validate: (value) => value.length > 0 || "API key is required",
      });
      if (!response.apiKey) {
        console.log(chalk.red("API key is required. Exiting..."));
        process.exit(1);
      }
      apiKey = response.apiKey;
    }

    if (!falKey) {
      const response = await prompts({
        type: "password",
        name: "falKey",
        message: "Enter your fal.ai API key (for image generation):",
        validate: (value) => value.length > 0 || "fal.ai API key is required",
      });
      if (!response.falKey) {
        console.log(chalk.red("fal.ai API key is required. Exiting..."));
        process.exit(1);
      }
      falKey = response.falKey;
    }

    if (!elevenlabsApiKey) {
      const response = await prompts({
        type: "password",
        name: "elevenlabsApiKey",
        message: "Enter your ElevenLabs API key:",
        validate: (value) =>
          value.length > 0 || "ElevenLabs API key is required",
      });
      if (!response.elevenlabsApiKey) {
        console.log(chalk.red("ElevenLabs API key is required. Exiting..."));
        process.exit(1);
      }
      elevenlabsApiKey = response.elevenlabsApiKey;
    }

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

    console.log(chalk.blue(`\n📖 Creating story: "${title}"`));
    console.log(chalk.blue(`📝 Topic: ${topic} | Provider: ${provider}\n`));

    const model = createModel(provider, apiKey!);

    const storyWithDetails: StoryMetadataWithDetails = {
      shortTitle: title!,
      content: [],
    };

    const storySpinner = ora("Generating story...").start();
    const storyRes = await structuredCompletion(
      getGenerateStoryPrompt(title!, topic!),
      StoryScript,
      model,
    );
    storySpinner.succeed(chalk.green("Story generated!"));

    const descriptionsSpinner = ora("Generating image descriptions...").start();
    const storyWithImagesRes = await structuredCompletion(
      getGenerateImageDescriptionPrompt(storyRes.text),
      StoryWithImages,
      model,
    );
    descriptionsSpinner.succeed(chalk.green("Image descriptions generated!"));

    for (const item of storyWithImagesRes.result) {
      const contentWithDetails: ContentItemWithDetails = {
        text: item.text,
        imageDescription: item.imageDescription,
        uid: uuidv4(),
        audioTimestamps: {
          characters: [],
          characterStartTimesSeconds: [],
          characterEndTimesSeconds: [],
        },
      };
      storyWithDetails.content.push(contentWithDetails);
    }

    const contentFs = new ContentFS(title!);
    contentFs.saveDescriptor(storyWithDetails);

    const imagesSpinner = ora("Generating images and voice...").start();
    const total = storyWithDetails.content.length * 2;

    for (let i = 0; i < storyWithDetails.content.length; i++) {
      const storyItem = storyWithDetails.content[i];

      imagesSpinner.text = `[${i * 2 + 1}/${total}] Generating image...`;
      await generateAiImage({
        prompt: storyItem.imageDescription,
        path: contentFs.getImagePath(storyItem.uid),
        falKey: falKey!,
        onRetry: (attempt) => {
          imagesSpinner.text = `[${i * 2 + 1}/${total}] Generating image (retry ${attempt})...`;
        },
      });

      imagesSpinner.text = `[${i * 2 + 2}/${total}] Generating voice...`;
      const timings = await generateVoice(
        storyItem.text,
        elevenlabsApiKey!,
        contentFs.getAudioPath(storyItem.uid),
      );
      storyItem.audioTimestamps = timings;
    }

    contentFs.saveDescriptor(storyWithDetails);
    imagesSpinner.succeed(chalk.green("Images and voice generated!"));

    const finalSpinner = ora("Building timeline...").start();
    const timeline = createTimeLineFromStoryWithDetails(storyWithDetails);
    contentFs.saveTimeline(timeline);
    finalSpinner.succeed(chalk.green("Timeline built!"));

    console.log(chalk.green.bold("\n✨ Story generation complete!\n"));
    console.log("Preview: " + chalk.blue("bun run dev"));
    console.log("Render:  " + chalk.blue("bun run render"));

    return {};
  } catch (error) {
    console.error(chalk.red("\n❌ Error:"), error);
    process.exit(1);
  }
}

const commonOptions = (yargs: ReturnType<typeof import("yargs")>) =>
  yargs
    .option("api-key", {
      alias: "k",
      type: "string",
      description: "AI provider API key (OpenAI or Anthropic)",
    })
    .option("fal-key", {
      alias: "f",
      type: "string",
      description: "fal.ai API key for image generation",
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
      choices: ["openai", "anthropic"] as const,
      description: "AI provider to use for story generation",
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
