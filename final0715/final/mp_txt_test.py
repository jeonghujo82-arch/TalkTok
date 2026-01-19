from flask import Flask, jsonify
import os
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from pydub import AudioSegment

def convert_to_wav(input_path, output_path):
    audio = AudioSegment.from_file(input_path)
    audio.export(output_path, format="wav")
app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Whisper 모델 준비
device = "cuda" if torch.cuda.is_available() else "cpu"
model_id = "openai/whisper-small"
processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id).to(device)

pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    device=0 if device == "cuda" else -1,  # device 인덱스 지정
    return_timestamps=True
)

# 텍스트 유사도 계산 함수
def calc_similarity(text1, text2):
    vectorizer = TfidfVectorizer().fit([text1, text2])
    vectors = vectorizer.transform([text1, text2])
    sim_score = cosine_similarity(vectors[0], vectors[1])[0][0]
    return round(sim_score * 100, 2)

@app.route('/test', methods=['GET'])
def test_whisper():
    # ✅ 테스트용 경로 지정 (수동으로 입력)
    audio_path = os.path.join(UPLOAD_FOLDER, 'sample_audio.m4a')  
    wav_path = os.path.join(UPLOAD_FOLDER, 'converted_audio.wav')  # 변환된 경로

    # 변환 수행
    convert_to_wav(audio_path, wav_path)# 직접 넣은 오디오 파일
    script_path = os.path.join(UPLOAD_FOLDER, 'sample_script.txt')    # 텍스트 대본


    # Whisper 처리
    result = pipe(wav_path)
    recognized_text = result['text']

    # 대본 불러오기
    with open(script_path, 'r', encoding='utf-8') as f:
        original_script = f.read()

    # 유사도 계산
    similarity = calc_similarity(recognized_text, original_script)

    return jsonify({
        "recognized_text": recognized_text,
        "similarity": f"{similarity}%"
    })

if __name__ == '__main__':
    app.run(debug=True)
    
    