# agents.md — reel-maker

## Purpose
CLI-driven AI video generator producing TikTok/Reels-style MP4s — CLI generates script + assets, Remotion renders the final video.

## Stack
- Framework: Remotion 4 (no traditional web framework)
- Language: TypeScript (Bun runtime for CLI)
- Styling: Remotion component styles (no Tailwind)
- DB: None (local filesystem — `descriptor.json` + media files in `public/content/`)
- Auth: None
- Testing: None
- Deploy: Remotion Studio (local preview), `remotion render` for output
- Package manager: pnpm

## Repo structure
```
cli/
  cli.ts            # Interactive story generator
  batch.ts          # Batch processing multiple stories
  reddit.ts         # Reddit post → video pipeline
  service.ts        # ElevenLabs TTS, fal.ai image gen, AI completions
  timeline.ts       # Converts story data + timestamps → timeline.json
modal/
  image_gen.py      # FLUX / Wan T2V/I2V / Kokoro TTS via Modal GPU
  latentsync.py     # Lip-sync / talking-head endpoint
src/
  Root.tsx          # Registers Remotion compositions, auto-discovers timelines
  components/
    AIVideo.tsx     # Main composition (intro + background + subtitles + audio)
    Background.tsx  # Slide backgrounds with blur transitions + scale animations
    Subtitle.tsx    # TikTok-style word captions (@remotion/captions)
    Word.tsx        # Individual word with spring animation + active highlight
  lib/
    types.ts        # Zod schemas + TS types for timeline
    constants.ts    # FPS=30, dimensions=1080×1920, intro duration
    utils.ts        # Frame timing, blur calc, file path helpers
public/content/<slug>/  # Generated per story (gitignored)
  descriptor.json       # Story metadata + audio timestamps (generation cache)
  timeline.json         # Remotion-consumable timeline
  images/, audio/, videos/
scripts/
  regen-timeline.ts # Regenerate timeline from existing assets
remotion.config.ts  # jpeg output, overwrite enabled
```

## Key commands
```bash
pnpm dev          # remotion studio (preview at localhost:3000)
pnpm gen          # bun cli/cli.ts — interactive story generator
pnpm batch        # bun cli/batch.ts — batch generate
pnpm reddit       # bun cli/reddit.ts — reddit source pipeline
pnpm render       # remotion render — render to video file
pnpm lint         # eslint + tsc
```

## Architecture notes
- **No web server.** Remotion Studio is the only "web" interface — a local preview tool.
- **Data flow**: CLI prompts → AI script → image gen + TTS per scene → `descriptor.json` cache → `timeline.ts` converts to `timeline.json` → Remotion renders MP4.
- **Timeline JSON is the API contract** between CLI and renderer. Three arrays: `Elements` (slide backgrounds + transitions), `Text` (word-level subtitle timestamps), `Audio`.
- **4 video modes**: `images` (static + zoom), `i2v` (image-to-video), `t2v` (text-to-video), `talking-head` (lip-synced avatar).
- **Caching**: `descriptor.json` and individual assets cached — regeneration skips existing files.
- **AI providers**: `@ai-sdk/anthropic`/openai/google for scripts; `@elevenlabs/elevenlabs-js` for TTS (word-level timestamps for caption sync); `@fal-ai/client` for images; `msedge-tts` as free TTS fallback.
- **Modal scripts** are standalone Python — GPU-heavy video gen (Wan2.1 T2V/I2V), TTS (Kokoro), lip-sync (LatentSync) run on Modal.com independently.
- **Video dimensions**: 1080×1920 (9:16 portrait, TikTok/Reels format).
- Husky pre-push hook runs lint.

## Active context
