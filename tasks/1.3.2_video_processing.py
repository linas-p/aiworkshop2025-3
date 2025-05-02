import cv2
import ollama
import os
import tempfile
from PIL import Image
import numpy as np
import textwrap
from unidecode import unidecode


def lithuanian_to_ascii_fast(text):
    trans_table = str.maketrans(
        'ąčęėįšųūžĄČĘĖĮŠŲŪŽ',
        'aceeisuuzACEEISUUZ'
    )
    return text.translate(trans_table)

# Config
MODEL_NAME = 'gemma3:12b'
VIDEO_PATH = 'video.mp4'
OUTPUT_VIDEO_PATH = 'annotated_video.mp4'
FRAME_INTERVAL = 50
SUMMARY_FRAME_COUNT = 400

frame_results = []

# Create temp dir for frames
temp_dir = tempfile.mkdtemp()

# Load video
cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS)
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps, (width * 2, height))

frame_count = 0

def get_frame_analysis(image_path):
    response = ollama.chat(
        model=MODEL_NAME,
        messages=[{
            'role': 'user',
            'content': 'Kas matoma paveikslėlyje? Gražinti atsakymą paprastu teksto formatu. Maksimaliai 4 sakiniai. Jei yra teksto jį gražinti atskirai.',
            'images': [image_path]
        }]
    )
    return response['message']['content']
text_img = np.zeros((height, width, 3), dtype=np.uint8)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    show_overlay = False
    text_output = ""

    blank_text_img = np.ones((height, width, 3), dtype=np.uint8) * 255
    combined = np.hstack((frame, blank_text_img))

    
    if frame_count % FRAME_INTERVAL == 0:
        # Save frame to file
        img_path = os.path.join(temp_dir, f'frame_{frame_count}.png')
        cv2.imwrite(img_path, frame)

        # Get analysis from Ollama
        text_output = get_frame_analysis(img_path)
        frame_results.append(text_output)

        # Create black background for text
        text_img = np.zeros((height, width, 3), dtype=np.uint8)

        # Wrap and draw text
        wrapped_lines = []
        for line in text_output.split('\n'):
            wrapped_lines.extend(textwrap.wrap(line, width=40))

        y0, dy = 50, 30
        for i, line in enumerate(wrapped_lines):
            y = y0 + i * dy
            line = lithuanian_to_ascii_fast(line)
            cv2.putText(text_img, line, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    combined = np.hstack((frame, text_img))

    out.write(combined)

cap.release()

# Analyze last 200 frame results
summary_prompt = "Apibendrink šiuos vaizdo įrašų kadrų aprašymus į trumpą santrauką:\n" + "\n".join(frame_results[-SUMMARY_FRAME_COUNT:])
summary_response = ollama.chat(
    model=MODEL_NAME,
    messages=[{
        'role': 'user',
        'content': summary_prompt
    }]
)

wrapped_lines = []
for line in summary_response['message']['content'].split('\n'):
    wrapped_lines.extend(textwrap.wrap(line, width=80))

# Display and save the summary frame
summary_img = np.ones((height, width * 2, 3), dtype=np.uint8) * 255
y0, dy = 50, 30
for i, line in enumerate(wrapped_lines):
    y = y0 + i * dy
    line = lithuanian_to_ascii_fast(line)
    #print(line)
    cv2.putText(summary_img, line, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
for _ in range(int(fps * 5)):  # Show summary for 5 seconds
    out.write(summary_img)

out.release()
print("New video with analysis saved to:", OUTPUT_VIDEO_PATH)