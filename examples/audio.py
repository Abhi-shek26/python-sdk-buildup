"""TTS + STT examples."""

from qubrid import QubridClient

client = QubridClient()

speech = client.audio.speech.create(
    model="qwen3-tts-flash",
    text="Today is a wonderful day to build something people love!",
    voice="Cherry",
    language_type="Auto",
)
print(speech.url)

transcript = client.audio.transcriptions.create(
    model="openai/whisper-large-v3",
    file="clip.wav",  # local path, file object, or (filename, bytes) tuple
)
print(transcript.text)
