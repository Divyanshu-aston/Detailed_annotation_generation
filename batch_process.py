import os

from image_processor import process_single_video


def process_video_folder(video_folder, output_folder, sample_rate=2):
    video_files = [
        f for f in os.listdir(video_folder)
        if f.endswith((".mp4", ".avi", ".mov", ".mkv"))
    ]

    for video_file in video_files:
        video_path = os.path.join(video_folder, video_file)

        output_name = os.path.splitext(video_file)[0] + "_features.npy"
        output_path = os.path.join(output_folder, output_name)

        process_single_video(
            video_path=video_path,
            output_path=output_path,
            sample_rate=sample_rate
        )