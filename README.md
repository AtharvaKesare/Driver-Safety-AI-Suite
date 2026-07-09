# Driver Safety AI Suite 🚗💤

[![Live Demo](https://img.shields.io/badge/Live_Demo-Play_Now-success?style=for-the-badge)](https://web-app-ten-amber.vercel.app)

An advanced, browser-based artificial intelligence suite designed to monitor driver fatigue and distraction in real-time. Built with JavaScript, MediaPipe, and HTML5 Canvas, this application runs entirely locally on the user's device (edge computing), ensuring 100% privacy and zero latency.

## Features ✨

- **Real-Time Drowsiness Detection**: Tracks the Eye Aspect Ratio (EAR) using 3D facial landmarks to detect if the driver's eyes are closing or drooping.
- **Yawn Detection**: Monitors the Mouth Aspect Ratio (MAR) to detect signs of early fatigue via yawning.
- **Distraction & Head Pose Monitoring**: Decomposes the facial transformation matrix into Euler angles (Yaw, Pitch, Roll) to determine if the driver's eyes and head are off the road for an extended period.
- **Live Analytics Dashboard**: A real-time `Chart.js` graph that tracks the driver's eye behavior over a 60-second rolling window.
- **Session Summaries & Safety Score**: At the end of a trip/session, it provides a comprehensive report of drowsy events, yawns, and distractions, calculating an overall Safety Score (0-100).
- **Responsive & Mobile Friendly**: Scales dynamically for mobile portrait and desktop widescreen webcams. Features vibrating haptic feedback on supported mobile devices.

## How It Works 🧠

The suite uses Google's `MediaPipe FaceLandmarker` model to extract 478 3D facial landmarks at 30-60 frames per second. 
1. **EAR (Eye Aspect Ratio)**: Computes the vertical distance between eyelids relative to horizontal width.
2. **MAR (Mouth Aspect Ratio)**: Computes the vertical distance of the inner lips relative to horizontal width.
3. **Head Pose**: Uses the 4x4 matrix decomposition to calculate real-world head rotation degrees.

## Usage 🚀

Since the AI model runs locally in the browser, no backend server is required. 

1. Clone the repository:
   ```bash
   git clone https://github.com/AtharvaKesare/Driver-Safety-AI-Suite.git
   ```
2. Navigate to the directory:
   ```bash
   cd Driver-Safety-AI-Suite
   ```
3. Run a local development server (e.g., using `npx serve` or Live Server):
   ```bash
   npx serve .
   ```
4. Open your browser to `http://localhost:3000`.

**Note:** Browsers require a secure context (HTTPS or Localhost) to access the webcam APIs.

## Privacy & Security 🔒
- **Zero Data Collection**: No video frames, images, or telemetry data are ever sent to a server.
- **Edge Computing**: The neural network weights are downloaded to your browser cache, and all inference happens on your local CPU/GPU.

## Tech Stack 🛠️
- Vanilla JavaScript
- HTML5 / CSS3
- [MediaPipe Tasks Vision](https://developers.google.com/mediapipe)
- [Chart.js](https://www.chartjs.org/)

## License 📄
This project is open-source and available for educational and non-commercial use.
