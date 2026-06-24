/**
 * build-lesson.ts — self-contained lesson-reel builder.
 *
 * For each scene: generate Edge-TTS narration (free MS neural voice) with
 * word-boundary timestamps, measure the clip with ffprobe, then concat all
 * clips (with short pauses) into one track. Scene + caption timings are exact
 * by construction — no forced alignment.
 *
 * Output: public/content/<slug>/{audio.mp3, lesson.json}
 */
import { MsEdgeTTS, OUTPUT_FORMAT } from "msedge-tts";
import { execFileSync } from "node:child_process";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { SCENES, LESSON_META, VOICE, PAUSE_SECONDS } from "./lesson-content";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(__dirname, "..");
const SLUG = LESSON_META.slug;
const OUT_DIR = path.join(ROOT, "public", "content", SLUG);
const TMP_DIR = path.join(ROOT, "out", "lesson-build");

type CaptionToken = { text: string; startMs: number; endMs: number };

async function ttsScene(text: string, outPath: string) {
  const tts = new MsEdgeTTS();
  await tts.setMetadata(VOICE, OUTPUT_FORMAT.AUDIO_24KHZ_48KBITRATE_MONO_MP3, {
    wordBoundaryEnabled: true,
  });
  const { audioStream, metadataStream } = tts.toStream(text);
  const audio: Buffer[] = [];
  const meta: Buffer[] = [];
  audioStream.on("data", (d: Buffer) => audio.push(d));
  metadataStream?.on("data", (d: Buffer) => meta.push(d));
  await new Promise<void>((res, rej) => {
    audioStream.on("close", res);
    audioStream.on("error", rej);
  });
  fs.writeFileSync(outPath, Buffer.concat(audio));

  // Word boundaries → caption tokens (ms, relative to this scene's start).
  type WB = { Type: string; Data: { Offset: number; Duration: number; text: { Text: string } } };
  const words: CaptionToken[] = meta
    .flatMap((b) => {
      try {
        return (JSON.parse(b.toString()).Metadata ?? []).filter((e: WB) => e.Type === "WordBoundary");
      } catch {
        return [];
      }
    })
    .map((e: WB) => ({
      text: e.Data.text.Text,
      startMs: e.Data.Offset / 10_000,
      endMs: (e.Data.Offset + e.Data.Duration) / 10_000,
    }));
  return words;
}

function probeDurationSeconds(file: string): number {
  const out = execFileSync("ffprobe", [
    "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", file,
  ]).toString().trim();
  return parseFloat(out);
}

async function main() {
  fs.mkdirSync(OUT_DIR, { recursive: true });
  fs.mkdirSync(TMP_DIR, { recursive: true });

  // Short silence pad reused between scenes.
  const silencePath = path.join(TMP_DIR, "pause.mp3");
  execFileSync("ffmpeg", [
    "-y", "-f", "lavfi", "-i", "anullsrc=r=24000:cl=mono",
    "-t", String(PAUSE_SECONDS), "-c:a", "libmp3lame", "-b:a", "48k", silencePath,
  ], { stdio: "ignore" });
  const pauseMs = PAUSE_SECONDS * 1000;

  const sceneClips: string[] = [];
  const timelineScenes: any[] = [];
  let cursorMs = 0;

  for (let i = 0; i < SCENES.length; i++) {
    const scene = SCENES[i];
    const clipPath = path.join(TMP_DIR, `scene-${i}.mp3`);
    process.stdout.write(`  scene ${i + 1}/${SCENES.length}: ${scene.visual.kind} … `);
    const words = await ttsScene(scene.narration, clipPath);
    const durMs = Math.round(probeDurationSeconds(clipPath) * 1000);
    console.log(`${(durMs / 1000).toFixed(2)}s, ${words.length} words`);

    const startMs = cursorMs;
    const captions: CaptionToken[] = words.map((w) => ({
      text: w.text,
      startMs: startMs + w.startMs,
      endMs: startMs + w.endMs,
    }));

    timelineScenes.push({
      ...scene,
      startMs,
      endMs: startMs + durMs,
      captions,
    });

    sceneClips.push(clipPath);
    cursorMs += durMs + pauseMs;
  }

  // Concat: scene, pause, scene, pause, … (drop trailing pause).
  const concatList = path.join(TMP_DIR, "concat.txt");
  const lines: string[] = [];
  sceneClips.forEach((clip, i) => {
    lines.push(`file '${clip}'`);
    if (i < sceneClips.length - 1) lines.push(`file '${silencePath}'`);
  });
  fs.writeFileSync(concatList, lines.join("\n"));
  const audioOut = path.join(OUT_DIR, "audio.mp3");
  execFileSync("ffmpeg", [
    "-y", "-f", "concat", "-safe", "0", "-i", concatList,
    "-c:a", "libmp3lame", "-b:a", "96k", audioOut,
  ], { stdio: "ignore" });

  const totalMs = cursorMs - pauseMs;
  const lesson = {
    slug: SLUG,
    title: LESSON_META.title,
    handle: LESSON_META.handle,
    audioUrl: "audio.mp3",
    totalMs,
    scenes: timelineScenes,
  };
  fs.writeFileSync(path.join(OUT_DIR, "lesson.json"), JSON.stringify(lesson, null, 2));

  console.log(`\n✓ built ${SLUG}: ${(totalMs / 1000).toFixed(1)}s, ${SCENES.length} scenes`);
  console.log(`  audio:  ${path.relative(ROOT, audioOut)}`);
  console.log(`  lesson: ${path.relative(ROOT, path.join(OUT_DIR, "lesson.json"))}`);
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
