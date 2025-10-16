import cv2
import numpy as np
from pathlib import Path

class NailoongFaceTracker:
    def __init__(self, nailoong_image_path):
        """Initialize the face tracker with a nailoong image."""
        self.nailoong_img = cv2.imread(nailoong_image_path, cv2.IMREAD_UNCHANGED)

        # If the image doesn't have an alpha channel, add one
        if self.nailoong_img.shape[2] == 3:
            # Create alpha channel (fully opaque)
            alpha_channel = np.ones((self.nailoong_img.shape[0], self.nailoong_img.shape[1], 1), dtype=self.nailoong_img.dtype) * 255
            self.nailoong_img = np.concatenate([self.nailoong_img, alpha_channel], axis=2)

        # Load the face cascade classifier
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        # Initialize video capture
        self.cap = cv2.VideoCapture(0)

    def overlay_image(self, background, overlay, x, y, w, h):
        """Overlay an image with transparency onto another image."""
        # Resize overlay to fit the face
        overlay_resized = cv2.resize(overlay, (w, h))

        # Ensure we don't go out of bounds
        if x < 0 or y < 0 or x + w > background.shape[1] or y + h > background.shape[0]:
            return background

        # Extract the region of interest
        roi = background[y:y+h, x:x+w]

        # Split the overlay into color and alpha channels
        overlay_bgr = overlay_resized[:, :, :3]
        overlay_alpha = overlay_resized[:, :, 3:4] / 255.0

        # Blend the images
        blended = (overlay_alpha * overlay_bgr + (1 - overlay_alpha) * roi).astype(np.uint8)

        # Place the blended region back into the background
        background[y:y+h, x:x+w] = blended

        return background

    def run(self):
        """Run the face tracking loop."""
        print("Face tracking started!")
        print("Press 's' to save a picture")
        print("Press 'q' to quit")

        saved_count = 0

        while True:
            ret, frame = self.cap.read()

            if not ret:
                print("Failed to grab frame")
                break

            # Convert to grayscale for face detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detect faces
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30)
            )

            # Draw rectangle and overlay nailoong on each face
            for (x, y, w, h) in faces:
                # Make nailoong slightly bigger than the face for effect
                scale_factor = 1.5
                new_w = int(w * scale_factor)
                new_h = int(h * scale_factor)
                new_x = x - (new_w - w) // 2
                new_y = y - (new_h - h) // 2

                frame = self.overlay_image(frame, self.nailoong_img, new_x, new_y, new_w, new_h)

            # Display the result
            cv2.imshow('Nailoong Face Tracker', frame)

            # Handle key presses
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break
            elif key == ord('s'):
                # Save the current frame
                output_dir = Path('output')
                output_dir.mkdir(exist_ok=True)
                filename = output_dir / f'nailoong_face_{saved_count:03d}.png'
                cv2.imwrite(str(filename), frame)
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

    # Get all image files
    image_extensions = ['.png', '.jpg', '.jpeg', '.webp']
    nailoong_images = [f for f in nailoong_dir.iterdir()
                       if f.suffix.lower() in image_extensions]

    if not nailoong_images:
        print("Error: No images found in 'nailoong' directory!")
        return

    # Display available images
    print("Available nailoong images:")
    for i, img in enumerate(nailoong_images):
        print(f"{i}: {img.name}")

    # Let user choose or use the first one
    choice = input(f"\nChoose image (0-{len(nailoong_images)-1}) or press Enter for first: ").strip()

    if choice == "":
        selected_image = nailoong_images[0]
    else:
        try:
            selected_image = nailoong_images[int(choice)]
        except (ValueError, IndexError):
            print("Invalid choice, using first image")
            selected_image = nailoong_images[0]

    print(f"\nUsing: {selected_image.name}")

    # Create and run tracker
    tracker = NailoongFaceTracker(str(selected_image))
    tracker.run()


if __name__ == "__main__":
    main()
