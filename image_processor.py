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

def extract_frames(video_path, sample_rate=2, timestamp=None, next_timestamp=None, resize=(224, 224), Sample_dir=None, save_clip=False, save_keyframes=False):
    """Extract frames from video at specified sample rate and time range."""
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

    writer = None

    if save_clip:

        clip_output_path = os.path.join(Sample_dir, f"clip_{timestamp:.2f}s_to_{next_timestamp:.2f}s.mp4")

        os.makedirs(os.path.dirname(clip_output_path), exist_ok=True)

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')

        writer = cv2.VideoWriter(
            clip_output_path,
            fourcc,
            fps,
            resize
        )

    # IMPORTANT: jump directly to start frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    frame_idx = start_frame

    while frame_idx < end_frame:
        ret, frame = cap.read()

        if not ret:
            break
        
        resized_frame = cv2.resize(frame, resize)

        # -----------------------------------
        # Save clip frame
        # -----------------------------------
        if writer is not None:
            writer.write(resized_frame)

        if (frame_idx - start_frame) % frame_interval == 0:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, resize)
            frames.append(frame)
            frame_indices.append(frame_idx)

            if save_keyframes:
                keyframe_dir = os.path.join(Sample_dir, "keyframes")
                os.makedirs(keyframe_dir, exist_ok=True)
                timestamp_sec = frame_idx / fps
                filename = (
                f"frame_{frame_idx}_"
                f"{timestamp_sec:.2f}s.jpg"
                                )
                save_path = os.path.join(keyframe_dir, filename)
                cv2.imwrite(save_path, resized_frame)

        frame_idx += 1

    cap.release()

    if writer is not None:
        writer.release()

    return np.array(frames), frame_indices


# -----------------------------
# 2. RGB Embeddings: DINOv2
# -----------------------------

def extract_rgb_embeddings(frames, Feature_dir=None):
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
    # saving numpy array of embeddings to disk for debugging
    np.save(os.path.join(Feature_dir, "rgb_embeddings.npy"), np.array(embeddings))

    return np.array(embeddings)


# -----------------------------
# 3. Optical Flow Features
# -----------------------------

def extract_optical_flow_features(frames, Feature_dir=None):
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
    
    # saving numpy array of flow features to disk for debugging
    np.save(os.path.join(Feature_dir, "flow_features.npy"), np.array(flow_features))

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

def extract_object_features(frames, Sample_dir=None, save_detected_frames=False, max_objects=10):
    """
    Extract compact object features per frame.

    Features:
    - number of objects
    - mean confidence
    - largest box area
    - object density

    Output shape: [num_frames, 4]
    """
    output_dir = os.path.join(Sample_dir, "detected_frames")

    if save_detected_frames:
        os.makedirs(output_dir, exist_ok=True)

    object_features = []

    for i, frame in enumerate(tqdm(frames, desc="Extracting object features")):
        results = yolo_model(frame, verbose=False)

        boxes = results[0].boxes

        if boxes is None or len(boxes) == 0:
            num_classes = len(yolo_model.names)
            feature_dim = 4 + num_classes

            object_features.append(np.zeros(feature_dim))

            if save_detected_frames:
                save_path = os.path.join(output_dir, f"frame_{i:04d}_no_objects.jpg")
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                cv2.imwrite(save_path, frame_bgr)

            continue

        confidences = boxes.conf.cpu().numpy()
        xyxy = boxes.xyxy.cpu().numpy()
        class_ids = boxes.cls.cpu().numpy().astype(int)

        # -----------------------------------
        # Confidence filtering
        # -----------------------------------
        confidence_threshold = 0.5

        valid_indices = confidences >= confidence_threshold

        confidences = confidences[valid_indices]
        xyxy = xyxy[valid_indices]
        class_ids = class_ids[valid_indices]

        if len(confidences) == 0:
            num_classes = len(yolo_model.names)
            feature_dim = 4 + num_classes

            object_features.append(np.zeros(feature_dim))

            continue

        class_hist = np.zeros(len(yolo_model.names))

        for cls_id in class_ids:
            class_hist[cls_id] += 1

        class_hist = class_hist / max_objects

        areas = []

        # Copy frame for drawing
        plotted_frame = frame.copy()

        for box, conf, cls_id in zip(xyxy, confidences, class_ids):
            x1, y1, x2, y2 = box
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

            area = max((x2 - x1), 0) * max((y2 - y1), 0)
            areas.append(area)

            class_name = yolo_model.names[cls_id]
            label = f"{class_name} {conf:.2f}"

            cv2.rectangle(
                plotted_frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            cv2.putText(
                plotted_frame,
                label,
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2
            )

        areas = np.array(areas)

        frame_area = frame.shape[0] * frame.shape[1]

        features = np.concatenate([
            np.array([
                len(boxes),
                np.mean(confidences),
                np.max(areas) / frame_area,
                len(boxes) / max_objects
            ]),
            class_hist
        ])

        object_features.append(features)

        if save_detected_frames:
            save_path = os.path.join(output_dir, f"frame_{i:04d}_detections.jpg")

            # Convert RGB to BGR before saving with OpenCV
            plotted_frame_bgr = cv2.cvtColor(plotted_frame, cv2.COLOR_RGB2BGR)
            cv2.imwrite(save_path, plotted_frame_bgr)

    os.makedirs(os.path.join(Sample_dir, "Features"), exist_ok=True)

    # saving numpy array of object features to disk for debugging
    np.save(os.path.join(Sample_dir, "Features/object_features.npy"), np.array(object_features))

    return np.stack(object_features)


# -----------------------------
# 6. Temporal Consistency Features
# -----------------------------

def extract_temporal_consistency_features(rgb_embeddings, Feature_dir=None):
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
    # saving numpy array of temporal features to disk for debugging
    np.save(os.path.join(Feature_dir, "temporal_features.npy"), np.array(temporal_features))

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

def process_single_video(video_path, output_path, timestamp, next_timestamp, sample_rate=2, Sample_dir=None):
    """
    Extract and merge Tier-1 features from a single video.
    """

    print(f"\nProcessing: {video_path}")

    frames, frame_indices = extract_frames(
        video_path,
        sample_rate=sample_rate,
        timestamp=timestamp,
        next_timestamp=next_timestamp,
        resize=(224, 224),
        Sample_dir=Sample_dir,
        save_clip=True, 
        save_keyframes=True
    )

    if len(frames) == 0:
        print("No frames extracted.")
        return None
    
    Feature_dir = os.path.join(Sample_dir, "Features")
    os.makedirs(Feature_dir, exist_ok=True)

    rgb_embeddings = extract_rgb_embeddings(frames, Feature_dir=Feature_dir)

    flow_features = extract_optical_flow_features(frames, Feature_dir=Feature_dir)

    # audio_features = extract_audio_mfcc_features(
    #     video_path,
    #     target_length=len(frames)
    # )

    object_features = extract_object_features(frames, Sample_dir=Sample_dir, save_detected_frames=True)

    temporal_features = extract_temporal_consistency_features(
        rgb_embeddings, Feature_dir=Feature_dir
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