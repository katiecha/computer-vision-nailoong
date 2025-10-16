import cv2
import numpy as np
from pathlib import Path
from deepface import DeepFace
import json

class ExpressionMatcher:
    def __init__(self, expression_mapping_file='nailoong_expressions.json'):
        """
        Initialize the expression matcher.

        Args:
            expression_mapping_file: JSON file mapping expressions to nailoong images
        """
        self.mapping_file = Path(expression_mapping_file)
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
        """Load the mapping of expressions to nailoong images."""
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
            nailoong_dir = Path('nailoong')
            if nailoong_dir.exists():
                image_extensions = ['.png', '.jpg', '.jpeg', '.webp']
                all_images = [str(f.name) for f in nailoong_dir.iterdir()
                             if f.suffix.lower() in image_extensions]

                # Put all images in neutral for now
                if all_images:
                    default_mapping["neutral"] = all_images

            with open(self.mapping_file, 'w') as f:
                json.dump(default_mapping, f, indent=2)

            print(f"Created default expression mapping file: {self.mapping_file}")
            print("Please edit this file to map your nailoong images to expressions!")

            return default_mapping

        with open(self.mapping_file, 'r') as f:
            return json.load(f)

    def get_nailoong_for_expression(self, expression):
        """Get a nailoong image for the given expression."""
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
            image_path = Path('nailoong') / image_name
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
        print("Make different facial expressions to see matching nailoong images!")
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

            # Get the corresponding nailoong image
            nailoong_img = self.get_nailoong_for_expression(self.current_expression)

            # Create display frame
            display_frame = frame.copy()

            # Display current expression
            cv2.putText(
                display_frame,
                f"Expression: {self.current_expression}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

            # Display the nailoong image if available
            if nailoong_img is not None:
                # Resize nailoong to take up more of the screen (2/3 of height)
                h, w = display_frame.shape[:2]
                nailoong_height = int(h * 0.66)  # Make it bigger - 2/3 of screen height
                aspect_ratio = nailoong_img.shape[1] / nailoong_img.shape[0]
                nailoong_width = int(nailoong_height * aspect_ratio)

                nailoong_resized = cv2.resize(nailoong_img, (nailoong_width, nailoong_height))

                # Place in top-right corner
                x_offset = w - nailoong_width - 10
                y_offset = 10

                display_frame[y_offset:y_offset+nailoong_height,
                             x_offset:x_offset+nailoong_width] = nailoong_resized

            # Show the frame
            cv2.imshow('Nailoong Expression Matcher', display_frame)

            # Handle key presses
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break
            elif key == ord('s'):
                # Save the current frame
                output_dir = Path('output')
                output_dir.mkdir(exist_ok=True)
                filename = output_dir / f'nailoong_expression_{saved_count:03d}.png'
                cv2.imwrite(str(filename), display_frame)
                print(f"Saved: {filename}")
                saved_count += 1

        # Cleanup
        self.cap.release()
        cv2.destroyAllWindows()


def main():
    # Check for nailoong images
    nailoong_dir = Path('nailoong')

    if not nailoong_dir.exists():
        print("Error: 'nailoong' directory not found!")
        return

    # Create and run matcher
    matcher = ExpressionMatcher()
    matcher.run()


if __name__ == "__main__":
    main()
