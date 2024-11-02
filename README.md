# YouTube Video Summarizer

This project summarizes YouTube video content using a combination of **audio transcription** (via Whisper) and **text summarization** (via T5-base). Users can input a YouTube URL and receive a brief summary of the video.

## Features

- **YouTube Audio Download**: Downloads audio from the provided YouTube video.
- **Transcription**: Converts audio to text using Whisper.
- **Summarization**: Summarizes the transcribed text using T5-base.
- **Web Interface**: Simple web interface for user input.

## Tech Stack

- **Python**: For backend and main functionalities.
- **Flask**: Backend framework to handle requests.
- **yt-dlp**: Downloads audio from YouTube.
- **Whisper**: Transcribes audio to text.
- **Transformers (T5-base)**: Summarizes the text.

## Project Structure

```
.
├── app.py               # Main Flask app for API endpoints
├── requirements.txt     # Dependencies for the project
└── templates
    └── index.html       # Frontend web interface
```

## Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/gv1999/youtube-video-summarizer.git
cd youtube-video-summarizer
```

### 2. Install Dependencies
Make sure you have Python installed, then install the required packages:
```bash
pip install -r requirements.txt
```

### 3. Run the Application
Start the Flask app to serve the summarization API:
```bash
python app.py
```
The app should now be running at `http://localhost:5000`.

### 4. Access the Web Interface
Open a web browser and go to `http://localhost:5000` to enter a YouTube URL and get a summary.

## Usage

1. Enter a YouTube URL in the input field.
2. Click “Submit” to retrieve and summarize the video content.
3. The summarized text will appear in the output section.

## Future Enhancements

- Error handling for invalid or inaccessible URLs.
- Progress indicators during download, transcription, and summarization.
- Customizable summary length options.
