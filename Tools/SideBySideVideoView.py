import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk

"""
This script demonstrates how to create a simple side-by-side video player using OpenCV and Tkinter.
The player allows you to load two video files and play them side-by-side.
"""

class VideoPlayer:
    def __init__(self, window, window_title, video_path_1, video_path_2):
        self.window = window
        self.window.title(window_title)

        # Load videos
        self.cap1 = cv2.VideoCapture(video_path_1)
        self.cap2 = cv2.VideoCapture(video_path_2)

        # Create canvas for displaying the video frames
        self.canvas = tk.Canvas(window, width=1280, height=480)
        self.canvas.pack()

        # Controls
        self.btn_play = tk.Button(window, text="Play", width=10, command=self.play)
        self.btn_play.pack(side=tk.LEFT)

        self.btn_pause = tk.Button(window, text="Pause", width=10, command=self.pause)
        self.btn_pause.pack(side=tk.LEFT)

        self.btn_stop = tk.Button(window, text="Stop", width=10, command=self.stop)
        self.btn_stop.pack(side=tk.LEFT)

        self.playing = False
        self.stopped = False

        self.delay = 15
        self.update()

        self.window.mainloop()

    def play(self):
        self.playing = True
        self.stopped = False

    def pause(self):
        self.playing = False

    def stop(self):
        self.playing = False
        self.stopped = True
        self.cap1.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self.cap2.set(cv2.CAP_PROP_POS_FRAMES, 0)

    def update(self):
        if self.playing and not self.stopped:
            ret1, frame1 = self.cap1.read()
            ret2, frame2 = self.cap2.read()

            if not ret1 or not ret2:
                self.stop()
                return

            # Resize frames
            frame1 = cv2.resize(frame1, (640, 480))
            frame2 = cv2.resize(frame2, (640, 480))

            # Concatenate frames side-by-side
            combined_frame = np.hstack((frame1, frame2))

            # Convert color space from BGR to RGB
            combined_frame = cv2.cvtColor(combined_frame, cv2.COLOR_BGR2RGB)

            # Convert the image to PIL format
            img = Image.fromarray(combined_frame)
            self.photo = ImageTk.PhotoImage(image=img)

            # Display on canvas
            self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)

        self.window.after(self.delay, self.update)

def open_files_and_play():
    file_paths = filedialog.askopenfilenames(filetypes=[("Video Files", "*.mp4 *.avi *.mov *.mkv")])
    if len(file_paths) == 2:
        root = tk.Tk()
        VideoPlayer(root, "Side-by-Side Video Player", file_paths[0], file_paths[1])

if __name__ == "__main__":
    open_files_and_play()
