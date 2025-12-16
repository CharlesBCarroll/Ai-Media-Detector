# main.py
from text_detector import detect_ai_text
from image_detector import detect_ai_image
from scraper import extract_text_from_url
from video_detector import detect_ai_video

def main():
    print("=== AI Media Detector CLI ===")
    print("1. Analyze text/article")
    print("2. Analyze image/photo")
    print("3. Analyze video")

    choice = input("Choose an option (1/2/3): ").strip()

    if choice == "1":
        user_input = input("Enter text or URL:\n> ").strip()
        if user_input.startswith("http"):
            text = extract_text_from_url(user_input)
            if not text:
                print("[!] Could not extract text from URL.")
                return
        else:
            text = user_input
        detect_ai_text(text)

    elif choice == "2":
        image_input = input("Enter local image path or image URL:\n> ").strip()
        detect_ai_image(image_input)
    elif choice == "3":
        video_input = input("Enter local video path:\n> ").strip()
        detect_ai_video(video_input)

    else:
        print("[!] Invalid option. Exiting.")

if __name__ == "__main__":
    main()
