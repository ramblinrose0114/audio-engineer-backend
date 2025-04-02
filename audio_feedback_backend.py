from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
import librosa
import soundfile as sf
import openai
import tempfile
import os

app = FastAPI()

# Allow CORS for frontend dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

openai.api_key = os.getenv("OPENAI_API_KEY")

@app.post("/upload")
async def upload_audio(
    file: UploadFile = File(...),
    lyrics: str = Form(""),
    chords: str = Form(""),
    vocal_notes: str = Form(""),
    goals: str = Form("")
):
    # Save uploaded file to a temp location
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        contents = await file.read()
        tmp.write(contents)
        tmp_path = tmp.name

    try:
        y, sr = librosa.load(tmp_path, sr=None)

        # Enhanced audio features
        duration = librosa.get_duration(y=y, sr=sr)
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        tempo = float(tempo)
        rms = librosa.feature.rms(y=y).mean()
        zcr = librosa.feature.zero_crossing_rate(y=y).mean()
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr).mean()
        dynamic_range = rms / librosa.feature.rms(y=y).min()
        mfcc = librosa.feature.mfcc(y=y, sr=sr).mean()

        print("Audio loaded. Duration:", duration)
        print("Tempo:", tempo)
        print("RMS:", rms)
        print("ZCR:", zcr)
        print("Spectral Centroid:", spectral_centroid)
        print("Dynamic Range:", dynamic_range)
        print("MFCC:", mfcc)

        analysis = (
            f"Duration: {duration:.2f}s, Tempo: {tempo:.1f} BPM, "
            f"RMS: {rms:.5f}, ZCR: {zcr:.5f}, "
            f"Spectral Centroid: {spectral_centroid:.2f}, "
            f"Dynamic Range: {dynamic_range:.2f}, MFCC: {mfcc:.2f}"
        )

        prompt = (
            "You are an expert audio engineer and music producer. Analyze the following audio features and artist-submitted context, "
            "and give specific, constructive feedback about how to improve the track. "
            "Include comments on mix/mastering, vocals, delivery, harmony, lyrics, and creative intent.\n\n"
            f"Audio Analysis:\n{analysis}\n\n"
            f"Lyrics:\n{lyrics}\n\n"
            f"Chord Progression:\n{chords}\n\n"
            f"Vocal Style / Delivery Notes:\n{vocal_notes}\n\n"
            f"Artist Goals:\n{goals}"
        )

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful audio engineer assistant."},
                {"role": "user", "content": prompt}
            ]
        )

        feedback = response.choices[0].message.content

        print("FEEDBACK:", feedback)
        print("ANALYSIS:", analysis)

        return {"feedback": feedback, "analysis": analysis}

    except Exception as e:
        print("ERROR:", e)
        return {"error": str(e)}

    finally:
        os.remove(tmp_path)

