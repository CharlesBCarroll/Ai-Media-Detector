# app.py
import os
from flask import Flask, render_template, request, redirect
from werkzeug.utils import secure_filename

from text_detector import detect_ai_text
from image_detector import detect_ai_image
from video_detector import detect_ai_video

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route("/")
def index():
    return render_template("index.html")


# --- TEXT ROUTE ---
@app.route("/analyze_text", methods=["POST"])
def analyze_text():
    user_input = request.form["text_input"]

    # URL or raw text handled by your backend
    error_msg = None
    try:
        result = detect_ai_text(user_input)
        if isinstance(result, dict):
            raw_score = result.get('final_score', 0.0)
            roberta_score = result.get('roberta', 0.0)
            gemini_score = result.get('gemini', 0.0)
            heuristic_score_val = result.get('heuristic', 0.0)
            reasoning_text = result.get('reasoning', '')
            is_ai_flag = result.get('is_ai', False)
        else:
            raw_score = result if result is not None else 0.0
            roberta_score = gemini_score = heuristic_score_val = 0.0
            reasoning_text = ''
            is_ai_flag = False
    except Exception as e:
        # Log the traceback to the server console for debugging
        import traceback
        traceback.print_exc()
        error_msg = str(e)
        raw_score = 0.0

    # Ensure a numeric percent value between 0 and 100
    try:
        score_percent = float(raw_score) * 100.0
    except Exception:
        score_percent = 0.0
    score_percent = max(0.0, min(100.0, score_percent))

    # Format component scores for display (percent strings)
    roberta_display = f"{roberta_score*100:.2f}%"
    gemini_display = f"{gemini_score*100:.2f}%"
    heuristic_display = f"{heuristic_score_val*100:.2f}%"

    return render_template("result_text.html",
                           text=user_input,
                           score=f"{score_percent:.2f}%",
                           score_num=score_percent,
                           roberta=roberta_display,
                           gemini=gemini_display,
                           heuristic=heuristic_display,
                           reasoning=reasoning_text,
                           is_ai=is_ai_flag,
                           error_msg=error_msg)

# --- IMAGE ROUTE ---
@app.route("/analyze_image", methods=["POST"])
def analyze_image():
    file = request.files["image_file"]
    if not file:
        return redirect("/")

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(filepath)

    try:
        result = detect_ai_image(filepath)
        if isinstance(result, dict):
            final = result.get('final_score', 0.0)
            clip_score = result.get('clip', 0.0)
            gemini_score = result.get('gemini', 0.0)
            reasoning_text = result.get('reasoning', '')
            is_ai_flag = result.get('is_ai', False)
        else:
            final = result if result is not None else 0.0
            clip_score = gemini_score = 0.0
            reasoning_text = ''
            is_ai_flag = False
    except Exception as e:
        import traceback
        traceback.print_exc()
        final = 0.0
        clip_score = gemini_score = 0.0
        reasoning_text = str(e)
        is_ai_flag = False

    final_percent = max(0.0, min(100.0, float(final) * 100.0))
    clip_display = f"{clip_score*100:.2f}%"
    gemini_display = f"{gemini_score*100:.2f}%"

    return render_template("result_image.html",
                       image_path=filepath,
                       score=f"{final_percent:.2f}%",
                       score_num=final_percent,
                       clip=clip_display,
                       gemini=gemini_display,
                       reasoning=reasoning_text,
                       is_ai=is_ai_flag)



# --- VIDEO ROUTE ---
@app.route("/analyze_video", methods=["POST"])
def analyze_video():
    file = request.files["video_file"]
    if not file:
        return redirect("/")

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(filepath)

    try:
        result = detect_ai_video(filepath)
        if isinstance(result, dict):
            avg = result.get('avg', 0.0)
            gemini_frame = result.get('gemini_frame', 0.0)
            reasoning_text = result.get('reasoning', '')
            is_ai_flag = result.get('is_ai', False)
        else:
            # legacy tuple support
            try:
                avg, gemini_frame, reasoning_text = result
            except Exception:
                avg = gemini_frame = 0.0
                reasoning_text = ''
            is_ai_flag = avg > 0.5
    except Exception as e:
        import traceback
        traceback.print_exc()
        avg = gemini_frame = 0.0
        reasoning_text = str(e)
        is_ai_flag = False

    avg_percent = max(0.0, min(100.0, float(avg) * 100.0))

    return render_template("result_video.html",
                       video_path=filepath,
                       avg_score=f"{avg_percent:.2f}%",
                       avg_num=avg_percent,
                       gemini_frame=f"{gemini_frame*100:.2f}%",
                       reasoning=reasoning_text,
                       is_ai=is_ai_flag)


if __name__ == "__main__":
    app.run(debug=True)
