import cv2
import numpy as np
from pathlib import Path
from deepface import DeepFace
import json

class ExpressionMatcher:
    def __init__(self, expression_mapping_file='expressions.json', image_directory='images'):
        """
        Initialize the expression matcher.

        Args:
            expression_mapping_file: JSON file mapping expressions to character images
            image_directory: Directory containing the character/expression images
        """
        self.mapping_file = Path(expression_mapping_file)
        self.image_directory = Path(image_directory)
        self.expression_images = self.load_expression_mapping()

        # Load face cascade for detection
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )

        # Initialize video capture
        self.cap = cv2.VideoCapture(0)

        # Cache for loaded images
        self.image_cache = {}

        # Current expression
        self.current_expression = "neutral"
        self.frame_count = 0

    def load_expression_mapping(self):
        """Load the mapping of expressions to character images."""
        if not self.mapping_file.exists():
            # Create a default mapping file
            default_mapping = {
                "happy": [],
                "sad": [],
                "angry": [],
                "neutral": [],
                "surprise": [],
                "fear": [],
                "disgust": []
            }

            # Try to auto-detect based on filenames
            if self.image_directory.exists():
                image_extensions = ['.png', '.jpg', '.jpeg', '.webp', '.gif', '.bmp']
                all_images = [str(f.name) for f in self.image_directory.iterdir()
                             if f.suffix.lower() in image_extensions]

                # Put all images in neutral for now
                if all_images:
                    default_mapping["neutral"] = all_images

            with open(self.mapping_file, 'w') as f:
                json.dump(default_mapping, f, indent=2)

            print(f"Created default expression mapping file: {self.mapping_file}")
            print("Please edit this file to map your images to expressions!")

            return default_mapping

        with open(self.mapping_file, 'r') as f:
            return json.load(f)

    def get_image_for_expression(self, expression):
        """Get a character image for the given expression."""
        expression = expression.lower()

        # Check if we have images for this expression
        if expression in self.expression_images and self.expression_images[expression]:
            image_name = self.expression_images[expression][0]
        elif self.expression_images.get("neutral"):
            # Fallback to neutral
            image_name = self.expression_images["neutral"][0]
        else:
            # Fallback to any available image
            for exp, images in self.expression_images.items():
                if images:
                    image_name = images[0]
                    break
            else:
                return None

        # Load from cache or disk
        if image_name not in self.image_cache:
            image_path = self.image_directory / image_name
            img = cv2.imread(str(image_path))
            if img is not None:
                self.image_cache[image_name] = img

        return self.image_cache.get(image_name)

    def detect_expression(self, frame):
        """Detect the dominant facial expression in the frame."""
        try:
            # Analyze the frame for emotions
            result = DeepFace.analyze(
                frame,
                actions=['emotion'],
                enforce_detection=False,
                detector_backend='opencv'
            )

            # Get the dominant emotion
            if isinstance(result, list):
                result = result[0]

            emotion = result['dominant_emotion']
            return emotion

        except Exception as e:
            # If detection fails, return previous expression
            return self.current_expression

    def run(self):
        """Run the expression matching loop."""
        print("Expression Matcher started!")
        print("Make different facial expressions to see matching images!")
        print("Press 's' to save a picture")
        print("Press 'q' to quit")
        print()

        saved_count = 0

        while True:
            ret, frame = self.cap.read()

            if not ret:
                print("Failed to grab frame")
                break

            # Only detect expression every 10 frames to improve performance
            if self.frame_count % 10 == 0:
                self.current_expression = self.detect_expression(frame)

            self.frame_count += 1

            # Get the corresponding character image
            char_img = self.get_image_for_expression(self.current_expression)

            # Create a cleaner UI with side-by-side view
            h, w, _ = frame.shape

            # Prepare character image
            if char_img is not None:
                # Resize image to match camera feed height
                aspect_ratio = char_img.shape[1] / char_img.shape[0]
                char_h = h
                char_w = int(char_h * aspect_ratio)
                char_resized = cv2.resize(char_img, (char_w, char_h))
            else:
                # Create a placeholder if no image is found
                char_h = h
                char_w = int(h * 0.75)  # Placeholder width
                char_resized = np.full((char_h, char_w, 3), (220, 220, 220), dtype=np.uint8)
                text = f"No image for '{self.current_expression}'"
                font = cv2.FONT_HERSHEY_SIMPLEX
                (text_w, text_h), _ = cv2.getTextSize(text, font, 0.8, 2)
                text_x = (char_w - text_w) // 2
                text_y = (char_h + text_h) // 2
                cv2.putText(char_resized, text, (text_x, text_y), font, 0.8, (100, 100, 100), 2)

            # Create the new layout with a bottom bar for text
            bar_height = 50
            total_width = w + char_w
            total_height = h + bar_height

            # Black canvas
            display_frame = np.zeros((total_height, total_width, 3), dtype=np.uint8)

            # Place camera feed and character image
            display_frame[0:h, 0:w] = frame
            display_frame[0:h, w:w + char_w] = char_resized

            # Add expression text on the bottom bar
            text = f"Expression: {self.current_expression.capitalize()}"
            cv2.putText(
                display_frame,
                text,
                (20, h + bar_height - 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2,
            )

            # Show the frame
            cv2.imshow('Expression Matcher', display_frame)

            # Handle key presses
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break
            elif key == ord('s'):
                # Save the current frame
                output_dir = Path('output')
                output_dir.mkdir(exist_ok=True)
                filename = output_dir / f'expression_{saved_count:03d}.png'
                cv2.imwrite(str(filename), display_frame)
                print(f"Saved: {filename}")
                saved_count += 1

        # Cleanup
        self.cap.release()
        cv2.destroyAllWindows()


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Expression Matcher - Match facial expressions to character images')
    parser.add_argument('--images', '-i', default='images',
                       help='Directory containing your character images (default: images)')
    parser.add_argument('--mapping', '-m', default='expressions.json',
                       help='JSON file mapping expressions to images (default: expressions.json)')

    args = parser.parse_args()

    # Check for image directory
    image_dir = Path(args.images)

    if not image_dir.exists():
        print(f"Error: Image directory '{args.images}' not found!")
        print(f"\nPlease create the directory and add your character images:")
        print(f"  mkdir {args.images}")
        print(f"  # Then add your .png, .jpg, .jpeg, or .webp files to {args.images}/")
        return

    # Check if directory has any images
    image_extensions = ['.png', '.jpg', '.jpeg', '.webp', '.gif', '.bmp']
    images = [f for f in image_dir.iterdir() if f.suffix.lower() in image_extensions]

    if not images:
        print(f"Error: No images found in '{args.images}' directory!")
        print(f"\nPlease add image files (.png, .jpg, .jpeg, .webp, etc.) to the directory.")
        return

    print(f"Found {len(images)} image(s) in '{args.images}' directory")
    print()

    # Create and run matcher
    matcher = ExpressionMatcher(
        expression_mapping_file=args.mapping,
        image_directory=args.images
    )
    matcher.run()


if __name__ == "__main__":
    main()
