# video_detector.py
import cv2
import numpy as np
from image_detector import detect_ai_image
from google import genai
import os
from PIL import Image
import requests

genai_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

def extract_keyframes(video_path, frame_interval=30, max_frames=15):
    """Extracts one frame every N frames and saves them temporarily."""
    vidcap = cv2.VideoCapture(video_path)
    count, saved, frames = 0, 0, []
    while vidcap.isOpened() and saved < max_frames:
        success, frame = vidcap.read()
        if not success:
            break
        if count % frame_interval == 0:
            frame_path = f"temp_frame_{saved}.jpg"
            cv2.imwrite(frame_path, frame)
            frames.append(frame_path)
            saved += 1
        count += 1
    vidcap.release()
    return frames


def gemini_reason_about_frame(frame_path):
    """
    Use Gemini to provide both a numeric probability (0–1) and reasoning text
    for a single image frame.
    """
    try:
        image = Image.open(frame_path)
        prompt = (
            "Examine this image and do two things:\n"
            "1. Give a number between 0 and 1 representing how likely it is "
            "that this image is AI-generated or a deepfake (0 = clearly real, 1 = clearly AI).\n"
            "2. Briefly explain your reasoning in one or two sentences."
        )
        resp = genai_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[prompt, image],
        )
        text = resp.text.strip()

        # Parse out the numeric value if possible
        score = 0.5
        for token in text.split():
            try:
                val = float(token)
                if 0 <= val <= 1:
                    score = val
                    break
            except ValueError:
                continue

        return score, text
    except Exception as e:
        print(f"[!] Gemini reasoning error: {e}")
        return 0.5, "Reasoning unavailable due to API error."


def detect_ai_video(video_path):
    """Run AI-generated likelihood detection on a video by sampling frames."""
    print("\n🎬 Analyzing video frames for deepfake / AI content...\n")

    frames = extract_keyframes(video_path)
    if not frames:
        print("Could not extract frames from video.")
        return {
            'final_score': 0.0,
            'avg': 0.0,
            'gemini_frame': 0.0,
            'reasoning': 'Could not extract frames from video.',
            'is_ai': False,
        }

    # Collect per-frame numeric scores and any per-frame reasoning if available
    frame_scores = []
    frame_details = []
    for i, frame in enumerate(frames):
        print(f"Analyzing frame {i+1}/{len(frames)}...")
        res = detect_ai_image(frame)
        if isinstance(res, dict):
            score = res.get('final_score', 0.0)
            fr_reason = res.get('reasoning', '')
            fr_gemini = res.get('gemini', None)
        else:
            score = res if res is not None else 0.0
            fr_reason = ''
            fr_gemini = None
        frame_scores.append(score)
        frame_details.append({
            'frame': frame,
            'score': score,
            'reasoning': fr_reason,
            'gemini': fr_gemini,
            'index': i,
        })

    avg_score = float(np.mean(frame_scores)) if frame_scores else 0.0
    print("\n--- VIDEO AI DETECTION SUMMARY ---")
    print(f"Average AI-likelihood across {len(frames)} frames: {avg_score*100:.2f}%")

    if avg_score > 0.5:
        print("⚠️  Video likely AI-generated or deepfake.")
    else:
        print("✅  Video likely authentic.")

    # Aggregate per-frame gemini scores when available
    gemini_vals = [d['gemini'] for d in frame_details if d.get('gemini') is not None]
    gemini_frame_mean = float(np.mean(gemini_vals)) if gemini_vals else 0.0

    # Build a fallback reasoning summary over all frames
    high_count = sum(1 for s in frame_scores if s > 0.5)
    total = len(frame_scores)
    top_frames = sorted(frame_details, key=lambda d: d['score'], reverse=True)[:3]
    examples = []
    for d in top_frames:
        short = d.get('reasoning') or ''
        examples.append(f"frame {d['index']+1}: {d['score']*100:.1f}% - {short[:120]}")

    fallback_reasoning = (
        f"Analyzed {total} frames. {high_count} frames ({(high_count/total*100) if total else 0:.1f}%) show strong AI indicators. "
        f"Average AI-likelihood across frames: {avg_score*100:.2f}%. Examples: " + "; ".join(examples)
    )

    # Try to synthesize an overall explanation using Gemini, passing the per-frame summary (not images)
    reasoning = fallback_reasoning
    try:
        prompt_parts = [
            "You are given per-frame AI-likelihood scores and short notes extracted from a video. \n",
            f"Total frames: {total}. Average score: {avg_score*100:.2f}%.\n",
            "Per-frame top examples:\n",
        ]
        for ex in examples:
            prompt_parts.append(ex + "\n")
        prompt_parts.append(
            "\nWrite a concise (2-3 sentence) explanation of whether the video is likely AI-generated or a deepfake, using the frame evidence. Mention the most important signals."
        )

        resp = genai_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=["".join(prompt_parts)],
        )
        gen_text = resp.text.strip()
        if gen_text:
            reasoning = gen_text + "\n\nComponent summary: " + fallback_reasoning
    except Exception as e:
        print(f"[!] Gemini video synthesis failed: {e}")

    result = {
        'final_score': avg_score,
        'avg': avg_score,
        'gemini_frame': gemini_frame_mean,
        'reasoning': reasoning,
        'is_ai': avg_score > 0.5,
        'frame_details': frame_details,
    }

    # Clean up temporary frame files created during extraction
    try:
        import os
        for f in frames:
            try:
                if os.path.exists(f):
                    os.remove(f)
            except Exception:
                pass
    except Exception:
        pass

    return result
