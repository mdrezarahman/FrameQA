import json
import os
import whisper
import numpy as np
import whisper_timestamped as whisper_ts

from ..common.config import Config

class AudioProcessor():
    def __init__(self) -> None:
        '''
        Analyze audio information 
        '''
        self.config = Config.getConfig()
        self.model = None
        self.mp3_file = "some.mp3"
        self.temp_dir = "Cache"

    def load_model(self, type: str | None, device: str = "cpu"):
        if type is None:
            # load default model using whisper-timestamped
            self.model = whisper_ts.load_model(self.config["model_whisper"], device=device)
        elif type == "base":
            self.model = whisper.load_model(type, device=device) # load base whisper model
        else:
            # this will change in future
            self.model = whisper_ts.load_model(self.config["model_whisper"], device=device)

    def process_audio(self, video_clip):
        # Extract the audio from the video clip
        audio_clip = video_clip.audio
        
        # Load the model
        self.load_model(type="whisper_timestamped", device="cuda")
        
        # Write the audio to a separate file
        self.temp_dir = os.path.join(os.getcwd(), self.temp_dir) # set filepath
        if not os.path.exists(self.temp_dir):
            os.makedirs(self.temp_dir)
        
        self.mp3_file = os.path.join(self.temp_dir, self.mp3_file)
        audio_clip.write_audiofile(self.mp3_file) #  saving audio_clip temporarily into an audio file 
        audio = whisper_ts.load_audio(self.mp3_file) # load the audio file
        
        audio_clip.close() # closing the audio stream

        result = whisper_ts.transcribe(self.model, audio, language="en")
        
        # Mark the timestamp along the text
        final_text = self.text_post_processing(result)

        # removing the audio file from cache dir
        if os.path.exists(self.mp3_file):
            os.remove(self.mp3_file)
        else:
            print("The file does not exist")

        return final_text

    def text_post_processing(self, result_dict: json):
        prompt = ""
        segments = result_dict["segments"]

        for i in range(len(segments)):
            prompt += "audio start position: " + str(segments[i]["start"]) + " sec\n"
            prompt += "corresponding text representation of the audio:" + segments[i]["text"] + "\n"
            prompt += "audio ending position: " + str(segments[i]["end"]) + "sec \n\n"
        
        return prompt
        
 