import subprocess
import os
import sys
import asyncio
import whisper
import datetime
import json
from deep_translator import GoogleTranslator
import edge_tts

LANGUAGE_MAPPINGS = {
    "hi": {"code": "hi-IN", "name": "Hindi"},
    "mr": {"code": "mr-IN", "name": "Marathi"},
    "ta": {"code": "ta-IN", "name": "Tamil"},
    "te": {"code": "te-IN", "name": "Telugu"},
    "gu": {"code": "gu-IN", "name": "Gujarati"},
    "bn": {"code": "bn-IN", "name": "Bengali"}
}
EDGE_TTS_VOICES = {
    "hi": "hi-IN-SwaraNeural",
    "mr": "mr-IN-AarohiNeural",
    "ta": "ta-IN-PallaviNeural",
    "te": "te-IN-ShrutiNeural",
    "bn": "bn-IN-TanishaaNeural",
    "gu": "gu-IN-DhwaniNeural",
}

if len(sys.argv) < 3:
    print("Error: Please provide input video file path and target language code.")
    sys.exit(1)

input_file = sys.argv[1]
target_language = sys.argv[2]

if target_language not in LANGUAGE_MAPPINGS:
    print(f"Error: Unsupported language code '{target_language}'. Supported codes: {', '.join(LANGUAGE_MAPPINGS.keys())}")
    sys.exit(1)

base_name = os.path.splitext(os.path.basename(input_file))[0]

def format_srt_time(seconds):
    td = datetime.timedelta(seconds=seconds)
    total_seconds = int(td.total_seconds())
    ms = int((seconds - int(seconds)) * 1000)
    hrs = total_seconds // 3600
    mins = (total_seconds % 3600) // 60
    secs = total_seconds % 60
    return f"{hrs:02}:{mins:02}:{secs:02},{ms:03}"

def translate_text(text, target_language):
    return GoogleTranslator(source='auto', target=target_language).translate(text)

def translate_subtitle(subtitle_path, target_language):
    translated_subtitle_path = f"translated_{base_name}_subtitle.srt"

    with open(subtitle_path, 'r', encoding='utf-8') as file:
        subtitle_content = file.readlines()

    translated_content = []

    for line in subtitle_content:
        stripped_line = line.strip()

        if (
            stripped_line
            and "-->" not in stripped_line
            and not stripped_line.isdigit()
        ):
            translated_line = translate_text(stripped_line, target_language) + "\n"
            translated_content.append(translated_line)
        else:
            translated_content.append(line)

    with open(translated_subtitle_path, 'w', encoding='utf-8') as file:
        file.writelines(translated_content)

    return translated_subtitle_path

def synthesize_text(text, output_file, target_language):
    voice = EDGE_TTS_VOICES[target_language]

    async def _run():
        communicate = edge_tts.Communicate(text, voice, rate='-10%')
        await communicate.save(output_file)

    asyncio.run(_run())

def get_duration(input_file):
    result = subprocess.run(
        [
            'ffprobe', '-v', 'error',
            '-show_entries', 'format=duration',
            '-of', 'default=noprint_wrappers=1:nokey=1',
            input_file
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    return float(result.stdout.strip())

def adjust_speed(input_file, output_file, speed_factor):
    if speed_factor < 0.5:
        factors = []
        remaining = speed_factor
        while remaining < 0.5:
            factors.append(0.5)
            remaining /= 0.5
        factors.append(round(remaining, 2))
        filter_str = "atempo=" + ",atempo=".join(map(str, factors))
    elif speed_factor > 2.0:
        factors = []
        remaining = speed_factor
        while remaining > 2.0:
            factors.append(2.0)
            remaining /= 2.0
        factors.append(round(remaining, 2))
        filter_str = "atempo=" + ",atempo=".join(map(str, factors))
    else:
        filter_str = f"atempo={speed_factor:.2f}"

    subprocess.run([
        'ffmpeg', '-hide_banner', '-loglevel', 'error',
        '-i', input_file,
        '-filter:a', filter_str,
        '-y', output_file
    ], check=True)

def generate_silence(duration, output_file):
    subprocess.run([
        'ffmpeg', '-hide_banner', '-loglevel', 'error',
        '-f', 'lavfi',
        '-i', 'anullsrc=channel_layout=stereo:sample_rate=44100',
        '-t', str(duration),
        '-q:a', '9',
        '-acodec', 'mp3',
        '-y', output_file
    ], check=True)

def parse_srt(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = [l.rstrip('\n') for l in f.readlines()]

    segments = []
    i = 0

    while i < len(lines):
        if lines[i].strip().isdigit():
            time_line = lines[i + 1].strip()
            start, end = time_line.split(' --> ')
            text_lines = []
            i += 2

            while i < len(lines) and lines[i].strip() != '':
                text_lines.append(lines[i].strip())
                i += 1

            text = " ".join(text_lines)

            segments.append({
                'start': start,
                'end': end,
                'text': text
            })
        i += 1

    return segments

def time_to_seconds(time_str):
    time_str = time_str.replace(',', '.')
    parts = time_str.split(':')
    if len(parts) == 3:
        return float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])
    elif len(parts) == 2:
        return float(parts[0]) * 60 + float(parts[1])
    else:
        return float(parts[0])

print(f"Processing video: {input_file}")
print(f"Target language: {LANGUAGE_MAPPINGS[target_language]['name']} ({target_language})")

# STEP 1: Transcription
print("Loading Whisper model...")
def load_whisper_safe(model_name="base"):
    try:
        return whisper.load_model(model_name)
    except RuntimeError as e:
        if "checksum" in str(e).lower():
            print("Corrupted model detected, deleting cache...")
            cache_path = os.path.join(os.path.expanduser("~"), ".cache", "whisper", f"{model_name}.pt")
            if os.path.exists(cache_path):
                os.remove(cache_path)
            print("Retrying download...")
            return whisper.load_model(model_name)
        else:
            raise
model = load_whisper_safe('base')
result = model.transcribe(input_file)

# STEP 2: Generate English subtitles
subtitle_file = f"{base_name}_subtitle.srt"
with open(subtitle_file, 'w', encoding='utf-8') as file:
    for idx, segment in enumerate(result['segments']):
        file.write(f"{idx + 1}\n")
        file.write(f"{format_srt_time(segment['start'])} --> {format_srt_time(segment['end'])}\n")
        file.write(f"{segment['text'].strip()}\n\n")

print("English subtitles generated.")

# STEP 3: Translate subtitles
translated_sub = translate_subtitle(subtitle_file, target_language)
print("Subtitles translated.")

# STEP 4: Parse translated subtitles
segments = parse_srt(translated_sub)

audio_files = []
previous_end = 0.0

for idx, seg in enumerate(segments):
    try:
        start = time_to_seconds(seg['start'])
        end = time_to_seconds(seg['end'])
        duration = end - start

        if duration <= 0:
            continue

        # Add silence before current segment
        if start > previous_end:
            silence_duration = start - previous_end
            silence_file = f"{base_name}_silence_{idx}.mp3"
            generate_silence(silence_duration, silence_file)
            audio_files.append(silence_file)

        # Generate TTS
        tts_file = f"{base_name}_segment_{idx}.mp3"
        synthesize_text(seg['text'], tts_file, target_language)

        tts_duration = get_duration(tts_file)

        speed_factor = tts_duration / duration
        if speed_factor < 0.1:
            speed_factor = 0.8
        elif speed_factor > 3.0:
            speed_factor = 2.0

        adjusted_file = f"{base_name}_adjusted_{idx}.mp3"
        adjust_speed(tts_file, adjusted_file, speed_factor)

        audio_files.append(adjusted_file)
        previous_end = end

        print(f"Processed segment {idx + 1}/{len(segments)}")

    except Exception as e:
        print(f"Error processing segment {idx}: {e}")
        continue

# STEP 5: Merge all audio
concat_file = f"{base_name}_concat.txt"
with open(concat_file, 'w', encoding='utf-8') as f:
    for afile in audio_files:
        f.write(f"file '{afile}'\n")

final_audio = f"{base_name}_final_audio.mp3"
subprocess.run([
    'ffmpeg', '-hide_banner', '-loglevel', 'error',
    '-f', 'concat',
    '-safe', '0',
    '-i', concat_file,
    '-c', 'copy',
    '-y', final_audio
], check=True)

# STEP 6: Merge final audio with video
output_video = f"{base_name}_output_video.mp4"
subprocess.run([
    'ffmpeg', '-hide_banner', '-loglevel', 'error',
    '-i', input_file,
    '-i', final_audio,
    '-map', '0:v:0',
    '-map', '1:a:0',
    '-c:v', 'copy',
    '-shortest',
    '-y', output_video
], check=True)

# STEP 7: Save language info
lang_info_file = f"{base_name}_lang_info.json"
with open(lang_info_file, 'w', encoding='utf-8') as f:
    json.dump({
        "language_code": target_language,
        "language_name": LANGUAGE_MAPPINGS[target_language]['name']
    }, f, ensure_ascii=False, indent=2)

# STEP 8: Cleanup temp files
temp_files = [subtitle_file, translated_sub, concat_file, final_audio]

for afile in audio_files:
    temp_files.append(afile)

for fpath in temp_files:
    if os.path.exists(fpath):
        try:
            os.remove(fpath)
        except Exception as e:
            print(f"Error deleting {fpath}: {e}")

print(f"Processing complete! Output video: {output_video}")