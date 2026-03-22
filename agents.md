# reel-maker

AI-powered TikTok/Reels video generator -- CLI generates script, images, voiceover, and timeline; Remotion renders the final video.

## Tech Stack

- **Runtime**: Bun
- **Video framework**: Remotion 4.0 (React-based programmatic video)
- **Language**: TypeScript (strict mode), Python for GPU endpoints
- **AI story generation**: Vercel AI SDK routed through a free gateway proxy with model fallback
- **Image generation**: FLUX models via HuggingFace (free), Stable Horde (free), Modal (self-hosted), or fal.ai (paid) -- auto-fallback chain
- **TTS**: Microsoft Edge TTS (`msedge-tts`, free, no key) as default; Modal Kokoro-82M as alternative
- **Video generation**: Wan2.1 T2V/I2V models on Modal; LatentSync for talking-head lip-sync
- **Schema validation**: Zod 4
- **GPU infra**: Modal (Python) for image gen, video gen, TTS, lip-sync

## Architecture

```
cli/                   # Bun CLI for content generation
  cli.ts               # Main entry -- yargs CLI, orchestrates full pipeline
  service.ts           # AI completions, image gen (multi-provider), TTS, video gen, lip-sync
  timeline.ts          # Converts descriptor.json -> timeline.json
  batch.ts             # Batch generation of multiple reels

src/                   # Remotion video rendering (React components)
  index.ts             # registerRoot
  Root.tsx             # Auto-discovers timelines from public/content/*/timeline.json
  components/
    AIVideo.tsx        # Main composition -- intro title + background + subtitles + audio
    Background.tsx     # Image/video backgrounds with blur transitions and scale animations
    Subtitle.tsx       # TikTok-style word captions using @remotion/captions
    Word.tsx           # Individual word with spring animation + active highlight
  lib/
    types.ts           # All Zod schemas and TS types
    constants.ts       # FPS=30, INTRO_DURATION=30frames, IMAGE_WIDTH=1024, IMAGE_HEIGHT=1792
    utils.ts           # Frame timing, blur calc, file path helpers

modal/                 # Modal GPU endpoints (Python)
  image_gen.py         # FLUX image gen, Wan T2V/I2V, Kokoro TTS, LatentSync
  latentsync.py        # Standalone LatentSync talking-head endpoint

public/content/<slug>/ # Generated content per story
  descriptor.json      # Story metadata with audio timestamps
  timeline.json        # Remotion-consumable timeline
  images/, audio/, videos/  # Generated assets

out/                   # Rendered .mp4 output files
```

### Data flow

1. CLI prompts for title+topic -> AI generates story script -> AI generates image descriptions
2. For each scene: generate image (multi-provider fallback) + generate voice (Edge TTS) + optionally generate video
3. All assets saved to `public/content/<slug>/` with `descriptor.json` as cache
4. `timeline.ts` converts descriptor into `timeline.json` with ms-level timings
5. Remotion reads `timeline.json` and renders final video

## Key Conventions

- **Caching**: descriptor.json and individual assets are cached -- regeneration skips existing
- **Video modes**: `images` (static with zoom), `i2v` (image-to-video), `t2v` (text-to-video), `talking-head` (lip-synced avatar)
- **Video dimensions**: 1080x1920 (9:16 portrait, TikTok/Reels format)
- **Types**: All timeline data validated with Zod schemas at both generation and render time
- **Provider fallback**: Image generation chains through configured providers with retry logic

## Commands

```bash
bun run dev                    # Open Remotion Studio
bun run gen                    # Interactive: generate content
bun run gen -- --title "X" --topic "Y"   # Non-interactive
bun run gen -- --video-style i2v         # Image-to-video mode
bun run gen -- --render                  # Generate + auto-render
bun run batch -- --topics "A, B, C"      # Multiple reels
bun run render -- <slug>       # Render specific story to out/<slug>.mp4
bun scripts/regen-timeline.ts  # Rebuild all timelines
bun run build                  # Bundle Remotion project
bun run lint                   # ESLint + tsc
```

## Environment Variables

```bash
# AI text generation
FREE_GATEWAY_API_KEY=          # API key for the free gateway proxy
FREE_GATEWAY_URL=              # Base URL for the gateway (OpenAI-compatible)

# Image generation (checked in order: HF > Stable Horde > Modal > fal)
HF_TOKEN=                      # HuggingFace -- FLUX.1-schnell, free tier
STABLE_HORDE_API_KEY=          # Stable Horde -- free with key
MODAL_IMAGE_GEN_URL=           # Modal self-hosted FLUX endpoint
FAL_KEY=                       # fal.ai -- $0.03/image

# Video generation (Modal, optional)
MODAL_T2V_URL=                 # Text-to-video (Wan2.1-T2V)
MODAL_I2V_URL=                 # Image-to-video (Wan2.1-I2V)
MODAL_LIPSYNC_URL=             # Talking-head (LatentSync)
```

## Current State

**Done:**
- Full pipeline: story generation -> image gen -> TTS -> timeline -> Remotion render
- Multiple image providers with automatic fallback
- Edge TTS with word-level timestamps for caption sync
- TikTok-style animated word captions
- Batch generation
- Video modes: static images, i2v, t2v, talking-head
- Caching (skips already-generated assets)
- Style customization via CLI flags
- Modal GPU endpoints for image/video/TTS/lipsync

**Not done:**
- No tests, no CI/CD
- No remote deployment of the Remotion renderer
- Modal endpoints require manual setup
