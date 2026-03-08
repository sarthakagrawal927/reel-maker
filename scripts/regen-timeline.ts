import * as fs from "fs";
import * as path from "path";
import { createTimeLineFromStoryWithDetails } from "../cli/timeline";
import type { StoryMetadataWithDetails } from "../src/lib/types";

const contentDir = path.join(process.cwd(), "public", "content");
const stories = fs.readdirSync(contentDir);

for (const story of stories) {
  const descriptorPath = path.join(contentDir, story, "descriptor.json");
  const timelinePath = path.join(contentDir, story, "timeline.json");

  if (!fs.existsSync(descriptorPath)) continue;

  const descriptor = JSON.parse(
    fs.readFileSync(descriptorPath, "utf-8"),
  ) as StoryMetadataWithDetails;

  const timeline = createTimeLineFromStoryWithDetails(descriptor);
  fs.writeFileSync(timelinePath, JSON.stringify(timeline, null, 2));
  console.log(`✓ Regenerated timeline for: ${story}`);
}
