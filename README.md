# Expression Matcher

## Overview

A real-time facial expression detection system that matches your expressions to character images. Detects 3 expressions: happy, sad, and angry.

## Requirements

- Python 3.8+
- Webcam
- Character images for expressions

## Installation

Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Add your character images to the `images/` directory

2. Edit `expressions.json` to map images to expressions:
```json
{
  "happy": ["happy.png"],
  "sad": ["sad.png"],
  "angry": ["angry.png"]
}
```

3. Run the program:
```bash
python expression_matcher.py
```

### Controls
- Make facial expressions to see matching images
- Press 's' to save a screenshot
- Press 'q' to quit

### Command Line Options
- `--images <dir>`: Specify image directory (default: images)
- `--mapping <file>`: Specify mapping file (default: expressions.json)

## Credits

Built with [OpenCV](https://opencv.org/), [DeepFace](https://github.com/serengil/deepface), and [TensorFlow](https://www.tensorflow.org/)