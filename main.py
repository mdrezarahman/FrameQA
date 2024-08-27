from moviepy.editor import VideoFileClip
from ai.providers.store.qdrant import Store
from ai.video_analyzer.audio_processor import AudioProcessor

import openai

def load():
    # test the whisper audio extraction process
    video_path = "E:/test_dataset/hackathon_dataset/TU Munich 1.mp4"
    video_clip = VideoFileClip(video_path)
    obj = AudioProcessor()
    final_text = obj.process_audio(video_clip=video_clip)
    print(final_text)


load()