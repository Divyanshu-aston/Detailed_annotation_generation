# pip install openai opencv-python tqdm pandas

import os
import pandas as pd
import cv2
import json
import base64
import numpy as np
from pathlib import Path
from tqdm import tqdm
from openai import OpenAI

client = OpenAI(api_key="OPENAI_API_KEY")


def encode_image_to_base64(image):
    _, buffer = cv2.imencode(".jpg", image)
    return base64.b64encode(buffer).decode("utf-8")


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

    if end_frame <= start_frame:
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
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)

    cap.release()
    return frames


CAPTION_PROMPT = """
You are creating high-quality pseudo-labels for training a video annotation model.

Given sampled frames from a short egocentric video segment and its Ego4D narration,
generate a detailed but factual annotation.

Do not invent details that are not visible or supported by the narration.

Return ONLY valid JSON with this structure:

{
  "summary": "...",
  "scene_context": "...",
  "visible_objects": ["..."],
  "human_actions": ["..."],
  "temporal_sequence": ["..."],
  "transcript_alignment": "...",
  "uncertain_details": ["..."],
  "detailed_caption": "..."
}
"""


def generate_caption_from_segment(
    video_path,
    start_time,
    end_time,
    narration,
    num_frames=8
):
    frames = sample_frames_from_segment(
        video_path=video_path,
        start_time=start_time,
        end_time=end_time,
        num_frames=num_frames
    )

    if len(frames) == 0:
        return None

    content = [
        {
            "type": "input_text",
            "text": (
                CAPTION_PROMPT
                + f"\n\nNarration:\n{narration}"
                + f"\n\nSegment time:\n{start_time:.2f}s to {end_time:.2f}s"
            )
        }
    ]

    for frame in frames:
        b64 = encode_image_to_base64(frame)

        content.append({
            "type": "input_image",
            "image_url": f"data:image/jpeg;base64,{b64}"
        })

    response = client.responses.create(
        model="gpt-4.1-mini",
        input=[
            {
                "role": "user",
                "content": content
            }
        ],
        temperature=0.2
    )

    return response.output_text


caption_output_folder = Path(
    r"C:\Users\250442661\ego4d_generated_captions"
)
caption_output_folder.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(r"C:\Users\250442661\Documents\Ego\ego4d_video_narration_matches.csv")

# Important: sort properly first
df = df.sort_values(
    by=["video_uid", "timestamp_sec"]
).reset_index(drop=True)


for i in tqdm(range(len(df) - 1)):

    row = df.iloc[i]
    next_row = df.iloc[i + 1]

    # Do not cross into a different video
    if row["video_uid"] != next_row["video_uid"]:
        continue

    video_uid = row["video_uid"]
    video_path = row["video_path"]

    start_time = float(row["timestamp_sec"])
    end_time = float(next_row["timestamp_sec"])

    narration = row["narration_text"]

    caption_json_text = generate_caption_from_segment(
        video_path=video_path,
        start_time=start_time,
        end_time=end_time,
        narration=narration,
        num_frames=8
    )

    if caption_json_text is None:
        continue

    output_path = caption_output_folder / f"{video_uid}_{i}.json"

    try:
        caption_data = json.loads(caption_json_text)
    except json.JSONDecodeError:
        caption_data = {
            "raw_output": caption_json_text
        }

    caption_data["video_uid"] = video_uid
    caption_data["video_path"] = video_path
    caption_data["start_time"] = start_time
    caption_data["end_time"] = end_time
    caption_data["narration_text"] = narration

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(caption_data, f, indent=4, ensure_ascii=False)