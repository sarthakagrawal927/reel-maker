#!/usr/bin/env node
/**
 * Batch reel generation.
 *
 * Usage:
 *   # Inline topics (comma-separated):
 *   bun run batch --topics "Space Facts, Ocean Depths, Ancient Rome" --title-prefix "Quick: "
 *
 *   # From a JSON file:
 *   bun run batch --file topics.json
 *
 *   # topics.json format:
 *   [
 *     { "title": "Space Facts", "topic": "Amazing space exploration facts" },
 *     { "title": "Ocean Depths", "topic": "Mysteries of the deep ocean" }
 *   ]
 *
 * All style flags from `bun run gen` are also supported and apply to every reel:
 *   bun run batch --topics "A, B, C" --voice am_adam --music /path/bg.mp3 --render
 */

import yargs from "yargs";
import { hideBin } from "yargs/helpers";
import chalk from "chalk";
import * as fs from "fs";
import * as dotenv from "dotenv";
import { generateStory } from "./cli";

dotenv.config({ quiet: true });

interface BatchEntry {
  title: string;
  topic: string;
}

interface BatchOptions {
  topics?: string;
  file?: string;
  titlePrefix?: string;
  voice?: string;
  captionColor?: string;
  captionSize?: number;
  captionPosition?: "top" | "bottom" | "center";
  strokeWidth?: number;
  strokeColor?: string;
  combineMs?: number;
  music?: string;
  musicVolume?: number;
  render?: boolean;
  provider?: string;
  imageProvider?: "fal" | "modal";
}

async function runBatch(options: BatchOptions) {
  let entries: BatchEntry[] = [];

  if (options.file) {
    const raw = fs.readFileSync(options.file, "utf-8");
    entries = JSON.parse(raw) as BatchEntry[];
  } else if (options.topics) {
    entries = options.topics.split(",").map((t, i) => ({
      title: `${options.titlePrefix ?? ""}${t.trim()}`,
      topic: t.trim(),
    }));
  } else {
    console.error(chalk.red("Provide --topics or --file"));
    process.exit(1);
  }

  console.log(chalk.bold(`\n🎬 Batch generating ${entries.length} reel(s)\n`));

  const results: { title: string; status: "ok" | "error"; error?: string }[] = [];

  for (let i = 0; i < entries.length; i++) {
    const entry = entries[i];
    console.log(chalk.bold.blue(`\n[${i + 1}/${entries.length}] "${entry.title}"`));

    try {
      await generateStory({
        title: entry.title,
        topic: entry.topic,
        voice: options.voice,
        captionColor: options.captionColor,
        captionSize: options.captionSize,
        captionPosition: options.captionPosition,
        strokeWidth: options.strokeWidth,
        strokeColor: options.strokeColor,
        combineMs: options.combineMs,
        music: options.music,
        musicVolume: options.musicVolume,
        render: options.render,
        provider: options.provider as any,
        imageProvider: options.imageProvider,
      });
      results.push({ title: entry.title, status: "ok" });
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      results.push({ title: entry.title, status: "error", error: msg });
      console.error(chalk.red(`Failed: ${msg}`));
    }
  }

  console.log(chalk.bold("\n📊 Batch summary:"));
  for (const r of results) {
    const icon = r.status === "ok" ? chalk.green("✔") : chalk.red("✘");
    console.log(`  ${icon} ${r.title}${r.error ? chalk.gray(` — ${r.error}`) : ""}`);
  }
}

yargs(hideBin(process.argv))
  .option("topics", {
    type: "string",
    description: 'Comma-separated topics, e.g. "Space, Ocean, Rome"',
  })
  .option("file", {
    type: "string",
    description: "Path to a JSON file: [{title, topic}, ...]",
  })
  .option("title-prefix", {
    type: "string",
    description: 'Prefix added to auto-generated titles (when using --topics)',
    default: "",
  })
  .option("voice", { type: "string", description: "Kokoro voice ID" })
  .option("caption-color", { type: "string" })
  .option("caption-size", { type: "number" })
  .option("caption-position", {
    type: "string",
    choices: ["top", "bottom", "center"] as const,
  })
  .option("stroke-width", { type: "number" })
  .option("stroke-color", { type: "string" })
  .option("combine-ms", { type: "number" })
  .option("music", { type: "string" })
  .option("music-volume", { type: "number" })
  .option("render", { type: "boolean" })
  .option("provider", {
    type: "string",
    choices: ["openai", "anthropic", "google"] as const,
  })
  .option("image-provider", {
    type: "string",
    choices: ["fal", "modal"] as const,
  })
  .help()
  .parseAsync()
  .then((argv) => {
    runBatch({
      topics: argv.topics as string | undefined,
      file: argv.file as string | undefined,
      titlePrefix: argv["title-prefix"] as string,
      voice: argv.voice as string | undefined,
      captionColor: argv["caption-color"] as string | undefined,
      captionSize: argv["caption-size"] as number | undefined,
      captionPosition: argv["caption-position"] as "top" | "bottom" | "center" | undefined,
      strokeWidth: argv["stroke-width"] as number | undefined,
      strokeColor: argv["stroke-color"] as string | undefined,
      combineMs: argv["combine-ms"] as number | undefined,
      music: argv.music as string | undefined,
      musicVolume: argv["music-volume"] as number | undefined,
      render: argv.render as boolean | undefined,
      provider: argv.provider as string | undefined,
      imageProvider: argv["image-provider"] as "fal" | "modal" | undefined,
    });
  });
