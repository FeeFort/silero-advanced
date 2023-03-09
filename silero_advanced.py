import torch
import sounddevice as sd
import time
import re
from silero_advanced.numText import num2text

class synthesis:
    def __init__(self,
        language: str='ru',
        model_id: str='v3_1_ru',
        sample_rate: int=48000,
        speaker: str='baya',
        device: torch.device=torch.device('cpu')
    ):
        self.language = language
        self.model_id = model_id
        self.sample_rate = sample_rate
        self.speaker = speaker
        self.device = device

    def synthesis(self, text):
        model, example_text = torch.hub.load(repo_or_dir='snakers4/silero-models',
                                            model='silero_tts',
                                            language=self.language,
                                            speaker=self.model_id)
        model.to(self.device)

        numbers = re.findall(r'-?\d+\+?', text)
        if numbers != []:
            for i in numbers:
                numText = num2text(int(i))            
                text = text.replace(i, numText)

        text = text.replace("+", " плюс ")
        text = text.replace("*", "+")

        audio = model.apply_tts(text=text,
                                speaker=self.speaker,
                                sample_rate=self.sample_rate)

        return audio

    def synthesisAndPlay(self, text):
        model, example_text = torch.hub.load(repo_or_dir='snakers4/silero-models',
                                            model='silero_tts',
                                            language=self.language,
                                            speaker=self.model_id)
        model.to(self.device)

        numbers = re.findall(r'-?\d+\+?', text)
        if numbers != []:
            for i in numbers:
                numText = num2text(int(i))
                text = text.replace(i, numText)

        text = text.replace("+", " плюс ")
        text = text.replace("*", "+")
            
        audio = model.apply_tts(text=text,
                                speaker=self.speaker,
                                sample_rate=self.sample_rate)

        sd.play(audio, self.sample_rate)
        time.sleep(len(audio) / self.sample_rate)
        sd.stop()

    def synthesisAndSaveWav(self, text):
        model, example_text = torch.hub.load(repo_or_dir='snakers4/silero-models',
                                            model='silero_tts',
                                            language=self.language,
                                            speaker=self.model_id)
        model.to(self.device)

        numbers = re.findall(r'-?\d+\+?', text)
        if numbers != []:
            for i in numbers:
                numText = num2text(int(i))
                text = text.replace(i, numText)

        text = text.replace("+", " плюс ")
        text = text.replace("*", "+")

        audio = model.save_wav(text=text,
                            speaker=self.speaker,
                            sample_rate=self.sample_rate)