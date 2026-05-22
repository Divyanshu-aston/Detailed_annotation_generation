import cv2
import numpy as np


def sample_frames_from_segment(
    video_path,
    start_time,
    end_time,
    num_frames=8
):

    cap = cv2.VideoCapture(video_path)

    fps = cap.get(cv2.CAP_PROP_FPS)

    start_frame = int(start_time * fps)
    end_frame = int(end_time * fps)

    total_segment_frames = end_frame - start_frame

    if total_segment_frames <= 0:
        cap.release()
        return []

    frame_indices = np.linspace(
        start_frame,
        end_frame - 1,
        num_frames,
        dtype=int
    )

    frames = []

    for frame_idx in frame_indices:

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)

        success, frame = cap.read()

        if success:

            frame = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            frames.append(frame)

    cap.release()

    return frames