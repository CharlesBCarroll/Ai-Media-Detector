# image_detector.py
import os
import requests
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel
from google import genai
import time
import random

# Load CLIP model for image heuristic
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

genai_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

def detect_gemini_image(image_path_or_url):
    """Use Gemini to analyze the image with retries and fallback."""
    for attempt in range(3):
        try:
            if image_path_or_url.startswith("http"):
                image = Image.open(requests.get(image_path_or_url, stream=True).raw)
            else:
                image = Image.open(image_path_or_url)

            prompt = (
                "Rate from 0 to 1 how likely this image is AI-generated or a deepfake. "
                "Respond with only a number."
            )
            resp = genai_client.models.generate_content(
                model="gemini-2.5-flash",
                contents=[prompt, image],
            )
            raw = resp.text.strip()
            score = float(raw)
            return max(0.0, min(score, 1.0))
        except Exception as e:
            err_msg = str(e)
            print(f"[!] Gemini image attempt {attempt+1} failed: {err_msg}")
            if "503" in err_msg or "UNAVAILABLE" in err_msg:
                wait = random.uniform(2, 5)
                print(f"Waiting {wait:.1f}s before retry...")
                time.sleep(wait)
                continue
            return 0.5
    print("[!] Gemini service unavailable after retries; using fallback score.")
    return 0.5

def detect_ai_image(image_path_or_url):
    # Load image for CLIP
    if image_path_or_url.startswith("http"):
        image = Image.open(requests.get(image_path_or_url, stream=True).raw)
    else:
        image = Image.open(image_path_or_url)

    texts = [
        "a real photograph of a person",
        "an AI-generated image of a person",
        "a computer-generated landscape",
        "a real photo taken by a camera"
    ]
    inputs = clip_processor(text=texts, images=image, return_tensors="pt", padding=True)
    outputs = clip_model(**inputs)
    logits = outputs.logits_per_image
    probs = logits.softmax(dim=1)[0].tolist()

    ai_likelihood_clip = (probs[1] + probs[2]) / 2.0

    gemini_score = detect_gemini_image(image_path_or_url)

    final_score = (ai_likelihood_clip * 0.3) + (gemini_score * 0.7)

    print("\n--- IMAGE AI DETECTION ---")
    for txt, p in zip(texts, probs):
        print(f"{txt:45s} -> {p*100:.2f}%")
    print(f"\nCLIP AI-likelihood : {ai_likelihood_clip*100:.2f}%")
    print(f"Gemini score       : {gemini_score*100:.2f}%")
    print(f"Final AI-likelihood: {final_score*100:.2f}%")

    if final_score > 0.5:
        print("⚠️  Image likely AI-generated or deepfake.")
    else:
        print("✅  Image likely real.")
    # Build fallback reasoning summary
    fallback_reasoning = (
        f"CLIP-derived AI likelihood: {ai_likelihood_clip*100:.2f}% (weight 30%).\n"
        f"Gemini model: {gemini_score*100:.2f}% (weight 70%).\n"
        f"Combined final score: {final_score*100:.2f}% — {('likely AI-generated' if final_score>0.5 else 'likely real')}."
    )

    reasoning = fallback_reasoning
    try:
        prompt = (
            "Write a short (1-3 sentence) plain-language explanation of whether the following image "
            "is AI-generated or likely real, and list concrete visual signals that support the conclusion.\n\n"
        )
        resp = genai_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[prompt, Image.open(image_path_or_url) if not image_path_or_url.startswith('http') else Image.open(requests.get(image_path_or_url, stream=True).raw)],
        )
        gen_text = resp.text.strip()
        if gen_text:
            reasoning = gen_text + "\n\nComponent summary: " + fallback_reasoning
    except Exception as e:
        print(f"[!] Gemini image reasoning failed: {e}")

    result = {
        'final_score': final_score,
        'clip': ai_likelihood_clip,
        'gemini': gemini_score,
        'reasoning': reasoning,
        'is_ai': final_score > 0.5,
    }

    return result
