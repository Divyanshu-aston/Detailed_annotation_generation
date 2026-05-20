import os
import cv2
import torch
import librosa
import numpy as np
from tqdm import tqdm
from moviepy import VideoFileClip
from ultralytics import YOLO
from transformers import AutoImageProcessor, AutoModel


# -----------------------------
# Device
# -----------------------------

device = "cuda" if torch.cuda.is_available() else "cpu"


# -----------------------------
# Load Models
# -----------------------------

# RGB visual backbone
image_processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
rgb_model = AutoModel.from_pretrained("facebook/dinov2-base").to(device)
rgb_model.eval()

# Object detector
yolo_model = YOLO("yolov8n.pt")


# -----------------------------
# 1. Frame Extraction
# -----------------------------

def extract_frames(video_path, sample_rate=2, timestamp=None, next_timestamp=None, resize=(224, 224)):
    cap = cv2.VideoCapture(video_path)

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 25

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    start_frame = 0
    if timestamp is not None:
        start_frame = max(0, int(timestamp * fps))

    end_frame = total_frames
    if next_timestamp is not None:
        end_frame = min(total_frames, int(next_timestamp * fps))

    frame_interval = max(int(fps / sample_rate), 1)

    frames = []
    frame_indices = []

    # IMPORTANT: jump directly to start frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    frame_idx = start_frame

    while frame_idx < end_frame:
        ret, frame = cap.read()

        if not ret:
            break

        if (frame_idx - start_frame) % frame_interval == 0:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, resize)
            frames.append(frame)
            frame_indices.append(frame_idx)

        frame_idx += 1

    cap.release()

    return np.array(frames), frame_indices


# -----------------------------
# 2. RGB Embeddings: DINOv2
# -----------------------------

def extract_rgb_embeddings(frames):
    """
    Returns DINOv2 CLS embeddings.
    Shape: [num_frames, 768]
    """

    embeddings = []

    with torch.no_grad():
        for frame in tqdm(frames, desc="Extracting RGB embeddings"):
            inputs = image_processor(
                images=frame,
                return_tensors="pt"
            ).to(device)

            outputs = rgb_model(**inputs)

            cls_embedding = outputs.last_hidden_state[:, 0, :]

            embeddings.append(
                cls_embedding.squeeze(0).cpu().numpy()
            )

    return np.array(embeddings)


# -----------------------------
# 3. Optical Flow Features
# -----------------------------

def extract_optical_flow_features(frames):
    """
    Extract compact optical flow statistics.
    Shape: [num_frames, 4]

    Features:
    - mean flow x
    - mean flow y
    - mean magnitude
    - max magnitude
    """

    if len(frames) < 2:
        return np.zeros((len(frames), 4))

    flow_features = []

    prev_gray = cv2.cvtColor(frames[0], cv2.COLOR_RGB2GRAY)

    # First frame has no previous flow
    flow_features.append(np.zeros(4))

    for i in tqdm(range(1, len(frames)), desc="Extracting optical flow"):
        curr_gray = cv2.cvtColor(frames[i], cv2.COLOR_RGB2GRAY)

        flow = cv2.calcOpticalFlowFarneback(
            prev_gray,
            curr_gray,
            None,
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0
        )

        flow_x = flow[..., 0]
        flow_y = flow[..., 1]

        magnitude, angle = cv2.cartToPolar(flow_x, flow_y)

        features = np.array([
            np.mean(flow_x),
            np.mean(flow_y),
            np.mean(magnitude),
            np.max(magnitude)
        ])

        flow_features.append(features)

        prev_gray = curr_gray

    return np.array(flow_features)


# -----------------------------
# 4. Audio Extraction
# -----------------------------

def extract_audio_from_video(video_path, output_audio_path="temp_audio.wav"):
    """
    Extract audio from video.
    """

    try:
        clip = VideoFileClip(video_path)

        if clip.audio is None:
            return None

        clip.audio.write_audiofile(
            output_audio_path,
            logger=None
        )

        return output_audio_path

    except Exception as e:
        print(f"Audio extraction failed: {e}")
        return None


def extract_audio_mfcc_features(video_path, target_length):
    """
    Extract MFCC audio features and align them to number of video frames.

    Output shape: [num_frames, 40]
    """

    audio_path = extract_audio_from_video(video_path)

    if audio_path is None:
        return np.zeros((target_length, 40))

    try:
        y, sr = librosa.load(audio_path, sr=16000)

        mfcc = librosa.feature.mfcc(
            y=y,
            sr=sr,
            n_mfcc=40
        ).T

        # Resize temporal dimension to match video frames
        mfcc_resized = resize_sequence(mfcc, target_length)

        if os.path.exists(audio_path):
            os.remove(audio_path)

        return mfcc_resized

    except Exception as e:
        print(f"MFCC extraction failed: {e}")
        return np.zeros((target_length, 40))


# -----------------------------
# 5. Object Detection Features
# -----------------------------

def extract_object_features(frames, max_objects=10):
    """
    Extract compact object features per frame.

    Features:
    - number of objects
    - mean confidence
    - largest box area
    - object density

    Output shape: [num_frames, 4]
    """

    object_features = []

    for frame in tqdm(frames, desc="Extracting object features"):
        results = yolo_model(frame, verbose=False)

        boxes = results[0].boxes

        if boxes is None or len(boxes) == 0:
            object_features.append(np.zeros(4))
            continue

        confidences = boxes.conf.cpu().numpy()
        xyxy = boxes.xyxy.cpu().numpy()

        areas = []

        for box in xyxy:
            x1, y1, x2, y2 = box

            area = max((x2 - x1), 0) * max((y2 - y1), 0)
            areas.append(area)

        areas = np.array(areas)

        frame_area = frame.shape[0] * frame.shape[1]

        features = np.array([
            len(boxes),
            np.mean(confidences),
            np.max(areas) / frame_area,
            len(boxes) / max_objects
        ])

        object_features.append(features)

    return np.array(object_features)


# -----------------------------
# 6. Temporal Consistency Features
# -----------------------------

def extract_temporal_consistency_features(rgb_embeddings):
    """
    Measures frame-to-frame feature change.

    Output shape: [num_frames, 3]

    Features:
    - L2 distance from previous frame
    - cosine similarity with previous frame
    - embedding change magnitude
    """

    temporal_features = []

    temporal_features.append(np.zeros(3))

    for i in range(1, len(rgb_embeddings)):
        prev = rgb_embeddings[i - 1]
        curr = rgb_embeddings[i]

        l2_dist = np.linalg.norm(curr - prev)

        cosine_sim = np.dot(prev, curr) / (
            np.linalg.norm(prev) * np.linalg.norm(curr) + 1e-8
        )

        change_magnitude = np.mean(np.abs(curr - prev))

        temporal_features.append(
            np.array([
                l2_dist,
                cosine_sim,
                change_magnitude
            ])
        )

    return np.array(temporal_features)


# -----------------------------
# Utility: Resize Sequence
# -----------------------------

def resize_sequence(sequence, target_length):
    """
    Resize variable-length sequence to target length using linear interpolation.
    """

    if len(sequence) == target_length:
        return sequence

    if len(sequence) == 0:
        return np.zeros((target_length, sequence.shape[-1]))

    old_indices = np.linspace(0, 1, len(sequence))
    new_indices = np.linspace(0, 1, target_length)

    resized = []

    for dim in range(sequence.shape[1]):
        resized_dim = np.interp(
            new_indices,
            old_indices,
            sequence[:, dim]
        )

        resized.append(resized_dim)

    return np.stack(resized, axis=1)


# -----------------------------
# 7. Final Feature Merger
# -----------------------------

def merge_tier1_features(
    rgb_embeddings,
    flow_features,
    object_features,
    temporal_features
):
    """
    Merge all Tier-1 features frame-wise.

    Final shape:
    [num_frames, total_feature_dim]

    Feature dimensions:
    - RGB DINOv2: 768
    - Optical flow: 4
    - Audio MFCC: 40
    - Object features: 4
    - Temporal consistency: 3

    Total: 819
    """

    min_len = min(
        len(rgb_embeddings),
        len(flow_features),
        len(object_features),
        len(temporal_features)
    )

    merged_features = np.concatenate(
        [
            rgb_embeddings[:min_len],
            flow_features[:min_len],
            object_features[:min_len],
            temporal_features[:min_len]
        ],
        axis=1
    )

    return merged_features


# -----------------------------
# 8. Complete Single Video Pipeline
# -----------------------------

def process_single_video(video_path, output_path, timestamp, next_timestamp, sample_rate=2):
    """
    Extract and merge Tier-1 features from a single video.
    """

    print(f"\nProcessing: {video_path}")

    frames, frame_indices = extract_frames(
        video_path,
        sample_rate=sample_rate,
        timestamp=timestamp,
        next_timestamp=next_timestamp
    )

    if len(frames) == 0:
        print("No frames extracted.")
        return None

    rgb_embeddings = extract_rgb_embeddings(frames)

    flow_features = extract_optical_flow_features(frames)

    # audio_features = extract_audio_mfcc_features(
    #     video_path,
    #     target_length=len(frames)
    # )

    object_features = extract_object_features(frames)

    temporal_features = extract_temporal_consistency_features(
        rgb_embeddings
    )

    merged_features = merge_tier1_features(
        rgb_embeddings,
        flow_features,
        object_features,
        temporal_features
    )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    np.save(output_path, merged_features)

    print(f"Saved merged features to: {output_path}")
    print(f"Final feature shape: {merged_features.shape}")

    return merged_features