# 🥊Boxing Simulator🥊

An interactive mini-game in Python/OpenCV where you have to hit targets on the screen with your hands, detected in real time by your webcam using MediaPipe Pose (PoseNet model).  
Obstacles appear to make the game more challenging, and a custom head image is displayed live on your head!

## Features

- **Hand and head detection** with MediaPipe Pose
- **Targets to hit** with your hand (real-time scoring)
- **Dynamic obstacles** to avoid with your head
- **Custom head overlay** on the player (image/head.png)

## Screenshots

Stand up in front of your screen and hit the red circles:

![Game preview 1](image/illustration1.png)

Watch out! You also have to dodge the red zones:

![Game preview 2](image/illustration2.png)

## Installation

1. **Clone the repository**  
   ```bash
   git clone https://github.com/yourusername/boxing_simulator.git
   cd boxing_simulator
   ```

2. **Install dependencies**  
   ```bash
   pip install -r requirements.txt
   ```

3. **(Optional) Add a custom head image** in `image/head.png` to personalize your head (PNG with transparency recommended).

4. **(Optional) Add more head images in `heads/`** for other uses.

## Usage

```bash
python main.py
```

- Press `q` to quit the game.

## Project Structure

```
boxing_simulator/
├── image/
│   └── head.png            # Head image displayed on the player
├── main.py                 # Main script
├── requirements.txt        # Python dependencies
├── README.md               # This file
└── .gitignore
```

## Dependencies

- opencv-python
- mediapipe
- numpy

---

Project made for fun and to experiment with PoseNet features.