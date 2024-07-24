from youtube_transcript_api import YouTubeTranscriptApi

def get_youtube_transcript(video_id):
    transcript = YouTubeTranscriptApi.get_transcript(video_id)
    transcript_text = " ".join([entry['text'] for entry in transcript])
    return transcript_text

if __name__ == "__main__":
    video_id = "TX9qSaGXFyg" 
    transcript_text = get_youtube_transcript(video_id)
    with open("youtube_transcript.txt", "w") as file:
        file.write(transcript_text)
