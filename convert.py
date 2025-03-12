import argparse

import cv2
from PIL import Image
import os

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mp4", required=True, help="Name of the mp4 file")
    parser.add_argument("--gif", default=True, help="target gif file")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    # Define input MP4 file and output GIF file
    video_path = args.mp4
    output_gif = args.gif

    # Open video file
    cap = cv2.VideoCapture(video_path)

    frames = []
    fps = int(cap.get(cv2.CAP_PROP_FPS))  # Get FPS of the video
    frame_interval = fps // 10  # Reduce frame rate (adjust for smoother GIF)

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break  # End of video

        if frame_count % frame_interval == 0:  # Skip frames to reduce GIF size
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
            img = Image.fromarray(frame_rgb)
            frames.append(img)

        frame_count += 1

    cap.release()

    # Save frames as GIF
    if frames:
        frames[0].save(output_gif, save_all=True, append_images=frames[1:], duration=100, loop=0)
        print(f"GIF saved as {output_gif}")
    else:
        print("No frames extracted!")

