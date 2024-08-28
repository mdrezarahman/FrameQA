import torch
import numpy as np
from huggingface_hub import hf_hub_download
from transformers import VideoLlavaForConditionalGeneration, VideoLlavaProcessor, BitsAndBytesConfig

from ..common.config import Config

class ClipProcessor():
    def __init__(self) -> None:
        '''
        Analyze clip information 
        '''
        self.config = Config.getConfig()
        self.model = None
        self.prompt = ("Please generate a concise but detailed description for this video. "
                       "Ensure the description meticulously covers all visible elements. "
                       "Include details of any text, objects, people, colors, textures, and "
                       "spatial relationships. Highlight contrasts, interactions, and any "
                       "notable features that stand out. Avoid assumptions and focus only "
                       "on what is clearly observable in the image.")
        self.max_tokens = 1024
        self.start_timestamp = 0.0
        self.step_timestamp = 10.0
        self.load_model()

    def load_model(self):
        # specify how to quantize the model
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )

        self.model = VideoLlavaForConditionalGeneration.from_pretrained(
            "LanguageBind/Video-LLaVA-7B-hf", 
            quantization_config=quantization_config,
            device_map="auto")
        self.processor = VideoLlavaProcessor.from_pretrained("LanguageBind/Video-LLaVA-7B-hf")

    def process_clip(self, video_clip):
        result = []
        t = self.start_timestamp
        while (t < video_clip.duration):
            video = video_clip.subclip(t, min(t + self.step_timestamp, video_clip.duration))
            video = [frame.tolist() for frame in video.iter_frames()]

            inputs = self.processor(text=self.prompt, videos=video, return_tensors="pt")
            
            out = self.model.generate(**inputs, max_new_tokens=self.max_tokens)
            self.processor.batch_decode(out, skip_special_tokens=True, clean_up_tokenization_spaces=True)

            result.append({
                "start": t,
                "text": out,
                "end": min(t + self.step_timestamp, video_clip.duration)
            })
            t += self.step_timestamp
        
        # Mark the timestamp along the text
        final_text = self.text_post_processing(result)

        return final_text

    def text_post_processing(self, results):
        prompt = ""
        segments = results["segments"]

        for result in results:
            prompt += "video start position: " + str(result["start"]) + " sec\n"
            prompt += "corresponding text representation of the video:" + result["text"] + "\n"
            prompt += "video ending position: " + str(result["end"]) + "sec \n\n"
        
        return prompt
