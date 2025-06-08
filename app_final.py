# from flask import Flask, render_template, Response, jsonify
# import cv2
# import mediapipe as mp
# import numpy as np

# app = Flask(__name__)

# # Global counter variable
# counter = 0
# stage = None

# # Initialize mediapipe pose and drawing utilities
# mp_drawing = mp.solutions.drawing_utils
# mp_pose = mp.solutions.pose

# # Function to calculate angle between three points
# def calculate_angle(a, b, c):
#     a = np.array(a)
#     b = np.array(b)
#     c = np.array(c)

#     radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
#     angle = np.abs(radians * 180.0 / np.pi)

#     if angle > 180.0:
#         angle = 360 - angle

#     return angle

# # Video streaming generator
# def generate_frames():
#     global counter, stage  # Use global variables

#     cap = cv2.VideoCapture(0)

#     with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
#         while cap.isOpened():
#             ret, frame = cap.read()
#             if not ret:
#                 break

#             image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             image.flags.writeable = False
#             results = pose.process(image)

#             image.flags.writeable = True
#             image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

#             try:
#                 landmarks = results.pose_landmarks.landmark
#                 shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
#                             landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
#                 elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
#                          landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
#                 wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
#                          landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

#                 angle = calculate_angle(shoulder, elbow, wrist)

#                 cv2.putText(image, str(angle), tuple(np.multiply(elbow, [640, 480]).astype(int)),
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

#                 # Curl counter logic
#                 if angle > 160:
#                     stage = "down"
#                 if angle < 30 and stage == "down":
#                     stage = "up"
#                     counter += 1

#                 # Render counter
#                 # cv2.rectangle(image, (0, 0), (225, 73), (245, 117, 16), -1)
#                 # cv2.putText(image, 'REPS', (15, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
#                 #             (0, 0, 0), 1, cv2.LINE_AA)
#                 # cv2.putText(image, str(counter), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 2,
#                 #             (255, 255, 255), 2, cv2.LINE_AA)
#                 # cv2.putText(image, 'STAGE', (65, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
#                 #             (0, 0, 0), 1, cv2.LINE_AA)
#                 # cv2.putText(image, stage if stage else "", (60, 60), cv2.FONT_HERSHEY_SIMPLEX, 2,
#                 #             (255, 255, 255), 2, cv2.LINE_AA)

#             except Exception as e:
#                 print("Pose detection error:", e)

#             mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

#             ret, buffer = cv2.imencode('.jpg', image)
#             frame = buffer.tobytes()
#             yield (b'--frame\r\n'
#                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

#     cap.release()

# @app.route('/')
# def index():
#     return render_template('index_final.html')

# @app.route('/video_feed')
# def video_feed():
#     return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# @app.route('/count')
# def get_count():
#     return jsonify({'count': counter})

# if __name__ == "__main__":
#     app.run(debug=True, port=8080)

from flask import Flask, request, jsonify
import numpy as np
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

counter = 0
stage = None

def calculate_angle(a, b, c):
    """Calculate the angle between three points (shoulder, elbow, wrist)."""
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - \
              np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    return 360 - angle if angle > 180.0 else angle

@app.route('/process_keypoints', methods=['POST'])
def process_keypoints():
    global counter, stage

    if not request.is_json:
        return jsonify({'error': 'Invalid content type'}), 400

    try:
        data = request.get_json()
        keypoints = data.get('keypoints')

        if not keypoints or len(keypoints) != 17:
            return jsonify({'error': 'Invalid number of keypoints'}), 400

        # LEFT arm keypoints
        left_shoulder = keypoints[5][:2]
        left_elbow = keypoints[7][:2]
        left_wrist = keypoints[9][:2]

        # Compute elbow angle
        angle = calculate_angle(left_shoulder, left_elbow, left_wrist)

        # Curl counter logic
        if angle > 160:
            stage = "down"
        if angle < 30 and stage == "down":
            stage = "up"
            counter += 1

        return jsonify({
            "count": counter,
            "angle": round(angle, 2)
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
@app.route('/reset_count', methods=['POST'])
def reset_count():
    global counter, stage
    counter = 0
    stage = None
    return jsonify({'message': 'Counter reset to 0'})

if __name__ == '__main__':
    # Run with threaded=True for better performance with async calls from browser
    app.run(debug=True, threaded=True)
