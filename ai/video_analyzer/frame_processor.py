import json
import os
import numpy as np
import ollama
import cv2

from ..common.config import Config

class FrameProcessor():
    def __init__(self) -> None:
        '''
        Analyze frame information 
        '''
        self.start_timestamp = 0.0
        self.step_timestamp = 10
        self.config = Config.getConfig()
        self.model = self.config.get('model', "llava")
        self.prompt = ("Please generate a concise but detailed description for this image. "
                       "Ensure the description meticulously covers all visible elements. "
                       "Include details of any text, objects, people, colors, textures, and "
                       "spatial relationships. Highlight contrasts, interactions, and any "
                       "notable features that stand out. Avoid assumptions and focus only "
                       "on what is clearly observable in the image.")
    
    def process_frames(self, video_clip):
        result = []
        t = self.start_timestamp
        self.step_timestamp = video_clip.duration // 2 # temporarily support only 2 frames per video 
        while (t <= video_clip.duration):
            image = video_clip.make_frame(t)
            success, encoded_image = cv2.imencode('.png', image)
            bytes_image = encoded_image.tobytes()
           
            tmp_result = ollama.generate(
                model='llava',
                prompt=self.prompt,
                images=[bytes_image],
                stream=False,
                options={'temperature': 0.0}
            )['response']

            result.append({
                "timestamp": t,
                "text": tmp_result,
            })
            t += self.step_timestamp
        
        # Mark the timestamp along the text
        final_text = self.text_post_processing(result)

        return final_text

    def text_post_processing(self, results):
        prompt = ""

        for result in results:
            prompt += "frame position: " + str(result["timestamp"]) + " sec\n"
            prompt += "corresponding text representation of the frame: " + result["text"] + "\n\n"
        
        return prompt
        
 