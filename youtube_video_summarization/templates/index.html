from flask import Flask, request, render_template, jsonify
import os
import yt_dlp
import whisper
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

app = Flask(__name__)

# Load Whisper and T5 models once at startup to avoid reloading them each time
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
whisper_model = whisper.load_model("base").to(device)
t5_model = T5ForConditionalGeneration.from_pretrained("t5-base").to(device)
t5_tokenizer = T5Tokenizer.from_pretrained("t5-base")

# Download audio from YouTube using yt-dlp
def download_audio(youtube_url, output_file='audio'):
    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{'key': 'FFmpegExtractAudio', 'preferredcodec': 'mp3', 'preferredquality': '192'}],
        'outtmpl': output_file,
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([youtube_url])
        mp3_file = f"{output_file}.mp3"
        if os.path.exists(mp3_file):
            return mp3_file
    except Exception as e:
        print(f"Error downloading audio: {e}")
    return None

# Check and download English subtitles if available
def download_subtitles(youtube_url):
    ydl_opts = {
        'format': 'bestaudio/best',
        'subtitles': 'en',
        'writesubtitles': True,
        'outtmpl': 'subtitles',
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([youtube_url])
        vtt_file = 'subtitles.en.vtt'
        if os.path.exists(vtt_file):
            return vtt_file
    except Exception as e:
        print(f"Error downloading subtitles: {e}")
    return None

# Transcribe audio using Whisper with GPU
def transcribe_audio(audio_file):
    with torch.no_grad():
        result = whisper_model.transcribe(audio_file)
        torch.cuda.empty_cache()  # Clear GPU memory after transcription
    return result['text']

# Extract text from subtitles
def extract_subtitles(vtt_file):
    with open(vtt_file, 'r') as file:
        subtitles = file.read()
    os.remove(vtt_file)  # Cleanup subtitle file
    return subtitles

# Split text into chunks for summarization
def split_text(text, chunk_size=512):
    sentences = text.split('. ')
    chunks, current_chunk = [], []
    for sentence in sentences:
        current_chunk.append(sentence)
        if len(' '.join(current_chunk)) > chunk_size:
            chunks.append(' '.join(current_chunk))
            current_chunk = []
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    return chunks

# Summarize text using preloaded T5 model on GPU with batch processing
def summarize_text_with_t5(text):
    chunks = split_text(text)
    summaries = []
    batch_size = 4  # Adjust batch size based on available GPU memory

    for i in range(0, len(chunks), batch_size):
        batch_chunks = chunks[i:i + batch_size]
        with torch.no_grad():
            # Encode all chunks in the batch at once
            inputs = t5_tokenizer(["summarize: " + chunk for chunk in batch_chunks], 
                                  return_tensors="pt", max_length=512, truncation=True, padding=True).to(device)
            summary_ids = t5_model.generate(inputs.input_ids, max_length=150, min_length=40, 
                                            length_penalty=2.0, num_beams=4, early_stopping=True)
            
            # Decode summaries and add to list
            batch_summaries = [t5_tokenizer.decode(ids, skip_special_tokens=True) for ids in summary_ids]
            summaries.extend(batch_summaries)
            
            # Clear GPU memory after processing each batch
            del inputs, summary_ids
            torch.cuda.empty_cache()

    # Return a more formal, newspaper-style summary
    formatted_summary = "Video Summarization:\n" + "\n".join([f"- {summary.strip()}" for summary in summaries])
    return formatted_summary

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/summarize', methods=['POST'])
def summarize():
    youtube_url = request.form['youtube_url']
    audio_file = download_audio(youtube_url)
    subtitle_file = download_subtitles(youtube_url)
    summary = None
    
    try:
        if subtitle_file:
            transcript = extract_subtitles(subtitle_file)
        elif audio_file:
            transcript = transcribe_audio(audio_file)
        else:
            raise Exception("No valid audio or subtitles found.")
        
        summary = summarize_text_with_t5(transcript)
    except Exception as e:
        print(f"Error processing video: {e}")
    finally:
        # Ensure cleanup of files after processing
        if audio_file and os.path.exists(audio_file):
            os.remove(audio_file)
        if subtitle_file and os.path.exists(subtitle_file):
            os.remove(subtitle_file)

    if summary:
        return jsonify({'summary': summary})
    return jsonify({'error': 'Failed to process the video'})

if __name__ == '__main__':
    app.run(debug=True)
