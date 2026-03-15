import {
  AbsoluteFill,
  Img,
  OffthreadVideo,
  staticFile,
  useCurrentFrame,
  useVideoConfig,
} from "remotion";
import { FPS, IMAGE_HEIGHT, IMAGE_WIDTH } from "../lib/constants";
import { BackgroundElement } from "../lib/types";
import { calculateBlur, getImagePath, getVideoPath } from "../lib/utils";

const EXTRA_SCALE = 0.2;

export const Background: React.FC<{
  item: BackgroundElement;
  project: string;
}> = ({ item, project }) => {
  const frame = useCurrentFrame();
  const localMs = (frame / FPS) * 1000;
  const { width, height } = useVideoConfig();

  const imageRatio = IMAGE_HEIGHT / IMAGE_WIDTH;
  const imgWidth = height;
  const imgHeight = imgWidth * imageRatio;

  let animScale = 1 + EXTRA_SCALE;
  const currentScaleAnim = item.animations?.find(
    (anim) =>
      anim.type === "scale" && anim.startMs <= localMs && anim.endMs >= localMs,
  );
  if (currentScaleAnim) {
    const progress =
      (localMs - currentScaleAnim.startMs) /
      (currentScaleAnim.endMs - currentScaleAnim.startMs);
    animScale =
      EXTRA_SCALE +
      progress * (currentScaleAnim.to - currentScaleAnim.from) +
      currentScaleAnim.from;
  }

  const imgScale = animScale;
  const top  = -(imgHeight * imgScale - height) / 2;
  const left = -(imgWidth  * imgScale - width)  / 2;

  const blur = calculateBlur({ item, localMs });
  const currentBlur = 25 * blur;

  const sharedStyle: React.CSSProperties = {
    width:    imgWidth  * imgScale,
    height:   imgHeight * imgScale,
    position: "absolute",
    top,
    left,
    filter:        `blur(${currentBlur}px)`,
    WebkitFilter:  `blur(${currentBlur}px)`,
    objectFit: "cover",
  };

  return (
    <AbsoluteFill>
      {item.videoUrl ? (
        <OffthreadVideo
          src={staticFile(getVideoPath(project, item.videoUrl))}
          style={sharedStyle}
          muted
        />
      ) : (
        <Img
          src={staticFile(getImagePath(project, item.imageUrl ?? ""))}
          style={sharedStyle}
        />
      )}
    </AbsoluteFill>
  );
};
