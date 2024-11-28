from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydub import AudioSegment
from io import BytesIO
import soundfile as sf
import whisper
import uvicorn

# Initialize FastAPI
app = FastAPI()

# Configure CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)

# Load the Whisper model
model = whisper.load_model("base")

@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    if file is None:
        raise HTTPException(status_code=400, detail="No file provided")
    try:
        # Read uploaded file
        audio_data = BytesIO(await file.read())
        
        # Convert audio file to wav format using pydub
        audio = AudioSegment.from_file(audio_data)
        audio = audio.set_channels(1)  # Ensure mono
        audio = audio.set_frame_rate(16000)  # Resample to 16kHz
        wav_data = BytesIO()
        audio.export(wav_data, format="wav")
        wav_data.seek(0)

        # Load the wav file as a NumPy array
        audio_array, _ = sf.read(wav_data, dtype="float32")

        # Transcribe using Whisper
        result = model.transcribe(audio_array, fp16=False, language="en")
        return {"text": result["text"]}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Khởi động server
if __name__ == "__main__":
  uvicorn.run(app, host="0.0.0.0", port=8000)