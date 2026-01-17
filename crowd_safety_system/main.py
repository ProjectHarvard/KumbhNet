"""
Main Entry Point

Runs the crowd safety pipeline end-to-end with:
- Live video display
- Density estimation
- Grid-based risk analysis
- Temporal smoothing for stable alerts
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
RISK_WINDOW_SIZE = 5          # Number of recent frames to consider
HIGH_RISK_CONFIRM_FRAMES = 3  # How many HIGHs needed to confirm danger


def main():
    video_path = "data/videos/crowd_input.mp4"

    # Store recent risk levels
    recent_risks = deque(maxlen=RISK_WINDOW_SIZE)

    try:
        for frame, frame_number in read_video_frames(video_path):
            try:
                # Convert BGR (OpenCV) → RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Step 1: Density estimation
                density_map = get_density_map(frame_rgb)

                # Step 2: Risk analysis
                risk_level, risk_score, risk_mask = analyze_risk(density_map)

                # Save raw risk into temporal buffer
                recent_risks.append(risk_level)

                # Step 3: Temporal smoothing
                high_count = recent_risks.count("HIGH")
                medium_count = recent_risks.count("MEDIUM")

                if high_count >= HIGH_RISK_CONFIRM_FRAMES:
                    final_risk = "HIGH"
                elif medium_count >= 2:
                    final_risk = "MEDIUM"
                else:
                    final_risk = "LOW"

                # Step 4: Generate heatmap (saved to disk)
                generate_heatmap(
                    density_map=density_map,
                    frame=frame_rgb,
                    frame_number=frame_number
                )

                # ---------------- LIVE DISPLAY ----------------

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

                cv2.imshow("Crowd Safety Monitor", display_frame)

                # Console logging (raw + final)
                print(
                    f"Frame {frame_number} | "
                    f"Raw Risk: {risk_level} ({risk_score:.3f}) | "
                    f"Final Risk: {final_risk}"
                )

                # Exit on 'q'
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

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
