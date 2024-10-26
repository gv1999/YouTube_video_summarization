# Required Libraries
!pip install yt-dlp torch torchvision torchaudio transformers
# Install required libraries
!pip install yt-dlp
!pip install git+https://github.com/openai/whisper.git
!pip install torch transformers
!apt-get install ffmpeg

import yt_dlp
import whisper
from transformers import pipeline, T5Tokenizer, T5ForConditionalGeneration

# Function to download audio from YouTube using yt-dlp
def download_audio(youtube_url):
    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'outtmpl': 'audio.%(ext)s',
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([youtube_url])
        return 'audio.mp3'  # The output file
    except Exception as e:
        print(f"Error downloading audio: {e}")
        return None

# Function to transcribe audio using Whisper
def transcribe_audio(audio_file):
    model = whisper.load_model("base")  # Choose appropriate model size
    result = model.transcribe(audio_file)
    return result['text']

# Function to split text into chunks for summarization
def split_text(text, chunk_size=512):
    sentences = text.split('. ')
    chunks = []
    current_chunk = []

    for sentence in sentences:
        current_chunk.append(sentence)
        if len(' '.join(current_chunk)) > chunk_size:
            chunks.append(' '.join(current_chunk))
            current_chunk = []

    if current_chunk:
        chunks.append(' '.join(current_chunk))

    return chunks

# Function to summarize text using T5-base
def summarize_text_with_t5(text):
    # Load T5 model and tokenizer
    model = T5ForConditionalGeneration.from_pretrained("t5-base")
    tokenizer = T5Tokenizer.from_pretrained("t5-base")

    chunks = split_text(text)
    summaries = []

    for chunk in chunks:
        input_ids = tokenizer.encode("summarize: " + chunk, return_tensors="pt", max_length=512, truncation=True)
        summary_ids = model.generate(input_ids, max_length=150, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=True)
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        summaries.append(summary)

    return ' '.join(summaries)

# Main function to summarize a YouTube video
def summarize_youtube_video(youtube_url):
    audio_file = download_audio(youtube_url)
    if audio_file is None:
        return "Failed to download audio."

    transcript = transcribe_audio(audio_file)
    summary = summarize_text_with_t5(transcript)
    return summary

# Example usage
youtube_url = 'https://youtu.be/fRJQ-I0hArU?si=KUd_U_bt76qw_10q'  # Replace with your YouTube video ID
summary = summarize_youtube_video(youtube_url)

print("Summary:", summary)
