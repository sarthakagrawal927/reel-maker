/**
 * lesson-content.ts — the JS closures lesson script + per-scene visuals.
 * Narration is tuned for ~50s at Edge-TTS pace. Visual specs are consumed by
 * the LessonReel Remotion composition.
 */

export const VOICE = "en-US-AriaNeural";
export const PAUSE_SECONDS = 0.22;

export const LESSON_META = {
  slug: "lesson-closures",
  title: "JavaScript Closures",
  handle: "@dailyjs",
};

// A code token: [text, role]. role drives the syntax color.
type Tok = [string, "kw" | "fn" | "var" | "num" | "punct" | "plain" | "ret" | "comment"];
export type CodeLine = Tok[];

export type Scene =
  | { narration: string; visual: { kind: "title"; headline: string; sub: string } }
  | { narration: string; visual: { kind: "concept"; lead: string; highlight: string; tail: string } }
  | { narration: string; visual: { kind: "code"; caption: string; lines: CodeLine[]; revealByLine: boolean; highlightLines?: number[] } }
  | { narration: string; visual: { kind: "scope"; aCounts: number[]; bCounts: number[] } }
  | { narration: string; visual: { kind: "recap"; line1: string; line2: string } }
  | { narration: string; visual: { kind: "cta"; handle: string; sub: string } };

const K = (s: string): Tok => [s, "kw"];
const F = (s: string): Tok => [s, "fn"];
const V = (s: string): Tok => [s, "var"];
const N = (s: string): Tok => [s, "num"];
const P = (s: string): Tok => [s, "punct"];
const _ = (s: string): Tok => [s, "plain"];
const R = (s: string): Tok => [s, "ret"];

const MAKE_COUNTER: CodeLine[] = [
  [K("function "), F("makeCounter"), P("() {")],
  [_("  "), K("let "), V("count"), P(" = "), N("0"), P(";")],
  [_("  "), R("return "), K("function"), P("() {")],
  [_("    "), V("count"), P("++;")],
  [_("    "), R("return "), V("count"), P(";")],
  [_("  "), P("};")],
  [P("}")],
];

const CALL_TWICE: CodeLine[] = [
  [K("const "), V("a"), P(" = "), F("makeCounter"), P("();")],
  [K("const "), V("b"), P(" = "), F("makeCounter"), P("();")],
  [_(""), ["// two independent counters", "comment"]],
];

export const SCENES: Scene[] = [
  {
    narration:
      "Why does this little counter remember its own number, even after the function that built it is long gone?",
    visual: { kind: "title", headline: "Closures", sub: "the trick behind functions that remember" },
  },
  {
    narration:
      "The answer is closures. A closure is simply a function bundled together with the scope it was born in.",
    visual: { kind: "concept", lead: "A closure is a function", highlight: "plus the scope", tail: "it was born in." },
  },
  {
    narration:
      "Here's makeCounter. It declares a count variable, then returns an inner function that bumps count up and hands it back.",
    visual: { kind: "code", caption: "makeCounter.js", lines: MAKE_COUNTER, revealByLine: true },
  },
  {
    narration:
      "Now watch what happens when we call makeCounter twice, into a and b.",
    visual: { kind: "code", caption: "two calls", lines: CALL_TWICE, revealByLine: true, highlightLines: [0, 1] },
  },
  {
    narration:
      "We get two separate counters. A climbs one, two, three, while b starts fresh at one. Neither can touch the other's count.",
    visual: { kind: "scope", aCounts: [1, 2, 3], bCounts: [1] },
  },
  {
    narration:
      "That's because every call to makeCounter creates a brand new scope, and the returned function keeps that scope alive.",
    visual: { kind: "concept", lead: "Every call makes a", highlight: "fresh, private scope", tail: "the inner function keeps alive." },
  },
  {
    narration:
      "So remember: a closure is the inner function, plus the variables it captured.",
    visual: { kind: "recap", line1: "closure  =  inner function", line2: "+  the variables it captured" },
  },
  {
    narration: "Follow for a new JavaScript concept every single day.",
    visual: { kind: "cta", handle: LESSON_META.handle, sub: "a new JS concept daily" },
  },
];
