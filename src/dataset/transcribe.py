import json
from elevenlabs import ElevenLabs
import httpx
from ..settings import settings

class ChunkedFileReader:
    def __init__(self, file_path, chunk_size=1024 * 1024):
        self.file = open(file_path, "rb")
        self.chunk_size = chunk_size

    def read(self, size=-1):
        # If no specific size is given, default to chunk_size
        if size == -1:
            size = self.chunk_size
        return self.file.read(size)

    def __iter__(self):
        # Provide an iterator interface for additional compatibility
        while True:
            chunk = self.read(self.chunk_size)
            if not chunk:
                break
            yield chunk

    def close(self):
        self.file.close()

def elevenlabs_stt(file_path: str):
    custom_httpx_client = httpx.Client(timeout=httpx.Timeout(120.0, read=120.0))
    client = ElevenLabs(api_key=settings.ELEVENLABS_API_KEY, httpx_client=custom_httpx_client)
    
    # Use our custom file-like object for chunked reading.
    audio_stream = ChunkedFileReader(file_path)
    
    response = client.speech_to_text.convert(
        model_id="scribe_v1",
        file=audio_stream,    # This now has a read() method
        language_code="urd",
        num_speakers=2,
        diarize=True
    )
    
    # Ensure the file is closed after the upload
    audio_stream.close()
    return response.json()

def main():
    audio_path = "src/dataset/audios/002 - Loose Talk Episode 46 - ARY Digital.webm"
    transcription_json = elevenlabs_stt(audio_path)
    with open("src/dataset/transcriptions/002-Episode 46.json", "w") as f:
        json.dump(transcription_json, f)

if __name__ == "__main__":
    main()
