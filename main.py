"""
Main Entry Point

Runs the crowd safety pipeline end-to-end with:
- Live video display (synchronized with inference)
- Density estimation
- Grid-based risk analysis
- Temporal smoothing to avoid false positives
"""

import sys
import cv2
from collections import deque

from core.video_reader import read_video_frames
from models.crowd_model import get_density_map
from core.risk_analyzer import analyze_risk
from core.heatmap_generator import generate_heatmap


# -------------------------------
# Temporal smoothing configuration
# -------------------------------
RISK_WINDOW_SIZE = 5
HIGH_RISK_CONFIRM_FRAMES = 3


def main():
    video_path = "data/videos/crowd_input.mp4"

    # Buffer to store recent risk levels
    recent_risks = deque(maxlen=RISK_WINDOW_SIZE)

    try:
        for frame, frame_number in read_video_frames(video_path):
            try:
                # ---------------- FRAME USED FOR EVERYTHING ----------------
                # Convert BGR (OpenCV) → RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # ---------------- INFERENCE ----------------
                density_map = get_density_map(frame_rgb)
                risk_level, risk_score, risk_mask = analyze_risk(density_map)

                # Store raw risk for temporal smoothing
                recent_risks.append(risk_level)

                # ---------------- TEMPORAL SMOOTHING ----------------
                high_count = recent_risks.count("HIGH")
                medium_count = recent_risks.count("MEDIUM")

                if high_count >= HIGH_RISK_CONFIRM_FRAMES:
                    final_risk = "HIGH"
                elif medium_count >= 2:
                    final_risk = "MEDIUM"
                else:
                    final_risk = "LOW"

                # ---------------- SAVE HEATMAP ----------------
                generate_heatmap(
                    density_map=density_map,
                    frame=frame_rgb,
                    frame_number=frame_number
                )

                # ---------------- LIVE DISPLAY (SYNCED) ----------------
                display_frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

                if final_risk == "HIGH":
                    color = (0, 0, 255)      # Red
                elif final_risk == "MEDIUM":
                    color = (0, 255, 255)   # Yellow
                else:
                    color = (0, 255, 0)     # Green

                cv2.putText(
                    display_frame,
                    f"Risk: {final_risk} | Score: {risk_score:.2f}",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    color,
                    2
                )

                cv2.putText(
                    display_frame,
                    f"Frame ID: {frame_number}",
                    (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2
                )

                # Show EXACT frame that was analyzed
                cv2.imshow("Crowd Safety Monitor", display_frame)

                # IMPORTANT: sync display to processing speed
                # This removes visual-risk mismatch
                if cv2.waitKey(30) & 0xFF == ord('q'):
                    break

                # ---------------- LOGGING ----------------
                print(
                    f"Frame {frame_number} | "
                    f"Raw Risk: {risk_level} ({risk_score:.3f}) | "
                    f"Final Risk: {final_risk}"
                )

            except Exception as e:
                print(f"Error processing frame {frame_number}: {e}", file=sys.stderr)
                continue

    except KeyboardInterrupt:
        print("\nPipeline interrupted by user.")
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
