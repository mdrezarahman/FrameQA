from moviepy.editor import VideoFileClip
from ai.providers.store.qdrant import Store
from ai.video_analyzer.audio_processor import AudioProcessor
from ai.video_analyzer.frame_processor import FrameProcessor

import os
import openai

class IndexingDriver():
    def __init__(self) -> None:
        self.folder_path = "E:/test_dataset/hackathon_dataset/"

    def indexing(self):
        '''
        Driver program to vectorize the video content
        '''
        load = Store()
        print(os.getcwd())

        # Traversing the test dataset directory
        
        obj_id = 1
        print(os.path.exists(self.folder_path))

        # Check if the folder path exists
        try:
            if os.path.exists(self.folder_path) and os.path.isdir(self.folder_path):
                # List all files in the folder
                file_list = os.listdir(self.folder_path)
                
                # Traverse each file in the folder 
                for file_name in file_list: 
                    file_path = os.path.join(self.folder_path, file_name)
                    
                    # Check if it's a file (not a subfolder)
                    if os.path.isfile(file_path):
                        try:
                            data = self.process_video(file_path)
                            load.add(objectId=obj_id, nodeId="Agg-01", parent=file_path, permissionId=-1, text=data, tables=-1)
                            obj_id += 1

                            # Open the file in read mode
                            # with open(file_path, 'r', encoding='utf8') as file:
                            #     # Perform operations with the file (e.g., read content)
                            #     data = file.read().replace('\n', '')
                            #     # load.begin(obj_id)
                            #     load.add(objectId=obj_id, nodeId="Agg-01", parent=file_path, permissionId=-1, text=data)
                            #     obj_id += 1

                        except Exception as e:
                            print(f"Error reading file {file_name}: {e}")
        except Exception as e:
            print(f"Path doesn't exist for the file {file_name}: {e}")


    def process_video(self, video_path):
        # test the whisper audio extraction process
        # video_path = "E:/test_dataset/hackathon_dataset/TU Munich 1.mp4"
        video_clip = VideoFileClip(video_path)
        audio_proc = AudioProcessor()
        frame_proc = FrameProcessor()
        audio_text = audio_proc.process_audio(video_clip=video_clip)
        frame_text = frame_proc.process_frames(video_clip=video_clip)
        final_text = audio_text + "\n\n" + frame_text
        print(audio_text)
        print("\n")
        print("\n")
        print("=========================================================")
        print("\n")
        print("\n")
        print(final_text)
        return final_text


if __name__ == "__main__":
    obj = IndexingDriver()
    obj.indexing()
