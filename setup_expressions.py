import json
from pathlib import Path
import cv2

def show_image_and_get_expression(image_path):
    """Display an image and ask user to categorize its expression."""
    img = cv2.imread(str(image_path))

    if img is None:
        print(f"Could not load {image_path}")
        return None

    # Resize if too large
    max_height = 600
    h, w = img.shape[:2]
    if h > max_height:
        scale = max_height / h
        new_w = int(w * scale)
        img = cv2.resize(img, (new_w, max_height))

    cv2.imshow('Nailoong Image - Press key to categorize', img)

    print(f"\nShowing: {image_path.name}")
    print("What expression does this nailoong have?")
    print("  1 - Happy")
    print("  2 - Sad")
    print("  3 - Angry")
    print("  4 - Neutral")
    print("  5 - Surprise")
    print("  6 - Fear")
    print("  7 - Disgust")
    print("  s - Skip this image")
    print("  q - Quit setup")

    while True:
        key = cv2.waitKey(0) & 0xFF

        if key == ord('1'):
            cv2.destroyAllWindows()
            return "happy"
        elif key == ord('2'):
            cv2.destroyAllWindows()
            return "sad"
        elif key == ord('3'):
            cv2.destroyAllWindows()
            return "angry"
        elif key == ord('4'):
            cv2.destroyAllWindows()
            return "neutral"
        elif key == ord('5'):
            cv2.destroyAllWindows()
            return "surprise"
        elif key == ord('6'):
            cv2.destroyAllWindows()
            return "fear"
        elif key == ord('7'):
            cv2.destroyAllWindows()
            return "disgust"
        elif key == ord('s'):
            cv2.destroyAllWindows()
            return "skip"
        elif key == ord('q'):
            cv2.destroyAllWindows()
            return "quit"


def main():
    nailoong_dir = Path('nailoong')

    if not nailoong_dir.exists():
        print("Error: 'nailoong' directory not found!")
        return

    # Get all image files
    image_extensions = ['.png', '.jpg', '.jpeg', '.webp']
    nailoong_images = [f for f in nailoong_dir.iterdir()
                       if f.suffix.lower() in image_extensions]

    if not nailoong_images:
        print("Error: No images found in 'nailoong' directory!")
        return

    # Initialize mapping
    expression_mapping = {
        "happy": [],
        "sad": [],
        "angry": [],
        "neutral": [],
        "surprise": [],
        "fear": [],
        "disgust": []
    }

    print("=" * 60)
    print("Nailoong Expression Setup")
    print("=" * 60)
    print(f"\nFound {len(nailoong_images)} images to categorize\n")

    for image_path in nailoong_images:
        expression = show_image_and_get_expression(image_path)

        if expression == "quit":
            print("\nQuitting setup...")
            break
        elif expression == "skip":
            print(f"Skipped {image_path.name}")
            continue
        elif expression:
            expression_mapping[expression].append(image_path.name)
            print(f"✓ Added {image_path.name} to '{expression}'")

    # Save the mapping
    output_file = Path('nailoong_expressions.json')
    with open(output_file, 'w') as f:
        json.dump(expression_mapping, f, indent=2)

    print(f"\n✓ Saved expression mapping to {output_file}")
    print("\nSummary:")
    for expression, images in expression_mapping.items():
        if images:
            print(f"  {expression}: {len(images)} image(s)")

    print("\nYou can now run: python expression_matcher.py")


if __name__ == "__main__":
    main()
