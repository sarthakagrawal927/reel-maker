import { Composition, getStaticFiles, staticFile } from "remotion";
import { AIVideo, aiVideoSchema } from "./components/AIVideo";
import { LessonReel, lessonSchema } from "./lesson/LessonReel";
import { FPS, INTRO_DURATION } from "./lib/constants";
import { getTimelinePath, loadTimelineFromFile } from "./lib/utils";

export const RemotionRoot: React.FC = () => {
  const staticFiles = getStaticFiles();
  const timelines = staticFiles
    .filter((file) => file.name.endsWith("timeline.json"))
    .map((file) => file.name.split("/")[1]);

  const lessons = staticFiles
    .filter((file) => file.name.endsWith("lesson.json"))
    .map((file) => file.name.split("/")[1]);

  return (
    <>
      {lessons.map((slug) => (
        <Composition
          key={`lesson-${slug}`}
          id={slug}
          component={LessonReel}
          fps={FPS}
          width={1080}
          height={1920}
          schema={lessonSchema}
          defaultProps={{ lesson: null }}
          calculateMetadata={async ({ props }) => {
            const res = await fetch(staticFile(`content/${slug}/lesson.json`));
            const lesson = await res.json();
            return {
              durationInFrames: Math.ceil((lesson.totalMs / 1000) * FPS) + FPS,
              props: { ...props, lesson },
            };
          }}
        />
      ))}
      {timelines.map((storyName) => (
        <Composition
          id={storyName}
          component={AIVideo}
          fps={FPS}
          width={1080}
          height={1920}
          schema={aiVideoSchema}
          defaultProps={{
            timeline: null,
          }}
          calculateMetadata={async ({ props }) => {
            const { lengthFrames, timeline } = await loadTimelineFromFile(
              getTimelinePath(storyName),
            );

            return {
              durationInFrames: lengthFrames + INTRO_DURATION,
              props: {
                ...props,
                timeline,
              },
            };
          }}
        />
      ))}
    </>
  );
};
