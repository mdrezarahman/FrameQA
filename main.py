from moviepy.editor import VideoFileClip
from ai.providers.store.qdrant import Store
from ai.video_analyzer.audio_processor import AudioProcessor
from ai.video_analyzer.frame_processor import FrameProcessor

import openai

def load():
    # test the whisper audio extraction process
    video_path = "E:/test_dataset/hackathon_dataset/TU Munich 1.mp4"
    video_clip = VideoFileClip(video_path)
    audio_proc = AudioProcessor()
    frame_proc = FrameProcessor()
    audio_text = audio_proc.process_audio(video_clip=video_clip)
    frame_text = frame_proc.process_frames(video_clip=video_clip)
    print(audio_text)
    print("\n")
    print("\n")
    print("=========================================================")
    print("\n")
    print("\n")
    print(frame_text)


load()