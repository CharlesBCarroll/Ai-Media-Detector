# text_detector.py
import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from google import genai
from analyzer import analyze_text_features, heuristic_score
import time
import random

tokenizer = AutoTokenizer.from_pretrained("roberta-base-openai-detector", use_fast=True)
model = AutoModelForSequenceClassification.from_pretrained("roberta-base-openai-detector")
genai_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

def detect_gemini_ai(text):
    """Ask Gemini to rate how likely text is AI-generated, with retries."""
    for attempt in range(3):  
        try:
            resp = genai_client.models.generate_content(
                model="gemini-2.5-flash",
                contents=[
                    "Respond with a number between 0 and 1 representing how likely this text is AI-generated:\n",
                    text
                ],
            )
            raw = resp.text.strip()
            score = float(raw)
            return max(0.0, min(score, 1.0))
        except Exception as e:
            err_msg = str(e)
            print(f"[!] Gemini API attempt {attempt+1} failed: {err_msg}")
            if "503" in err_msg or "UNAVAILABLE" in err_msg:
                # wait 2–5 seconds and retry
                wait = random.uniform(2, 5)
                print(f"Waiting {wait:.1f}s before retry...")
                time.sleep(wait)
                continue
            
            return 0.5
    print("[!] Gemini service still unavailable after retries; using fallback.")
    return 0.5

def detect_ai_text(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)

    roberta_ai_prob = probs[0][1].item()
    features = analyze_text_features(text)
    heuristic = heuristic_score(features)
    gemini_prob = detect_gemini_ai(text)

    final_score = (roberta_ai_prob * 0.6) + (gemini_prob * 0.25) + (heuristic * 0.15)

    print("\n--- TEXT AI DETECTION ---")
    print(f"RoBERTa AI prob        : {roberta_ai_prob*100:.2f}%")
    print(f"Gemini detection score : {gemini_prob*100:.2f}%")
    print(f"Heuristic indicator    : {heuristic*100:.2f}%")
    print(f"Final AI-likelihood    : {final_score*100:.2f}%")

    is_ai = final_score > 0.5
    if is_ai:
        print("⚠️  Text is likely AI-generated.")
    else:
        print("✅  Text appears human-written.")

    # Build a short fallback reasoning string summarizing component contributions
    fallback_reasoning = (
        f"RoBERTa model: {roberta_ai_prob*100:.2f}% (weight 60%).\n"
        f"Gemini model: {gemini_prob*100:.2f}% (weight 25%).\n"
        f"Heuristic signals: {heuristic*100:.2f}% (weight 15%).\n"
        f"Combined final score: {final_score*100:.2f}% — {('likely AI-generated' if is_ai else 'appears human-written')}.")

    # Ask Gemini to write a concise, text-specific explanation for the UI if possible.
    reasoning = fallback_reasoning
    try:
        prompt = (
            "Write a short (2-3 sentence) plain-language explanation of whether the following text "
            "is AI-generated or human-written and why. Mention concrete signals (style, repetition, phrasing, "
            "or other features) that support the conclusion. Be factual and non-judgmental.\n\nText:\n" + text + "\n\nAnswer:"
        )
        resp = genai_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[prompt],
        )
        gen_text = resp.text.strip()
        if gen_text:
            # Use the generated explanation but append component summary for transparency
            reasoning = gen_text + "\n\nComponent summary: " + fallback_reasoning
    except Exception as e:
        # If Gemini fails, keep the fallback reasoning and log the error
        print(f"[!] Gemini reasoning generation failed: {e}")

    # Return a structured result so the web UI can display component scores and reasoning
    return {
        'final_score': final_score,
        'roberta': roberta_ai_prob,
        'gemini': gemini_prob,
        'heuristic': heuristic,
        'reasoning': reasoning,
        'is_ai': is_ai,
    }
