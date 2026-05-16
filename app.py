import os
import cv2
import numpy as np
import mediapipe as mp
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from flask import Flask, render_template, request

# ======================
# CONFIG
# ======================
MODEL_WEIGHTS = "model/light_stgcn.pth"
CLASSES_NPY = "model/classes.npy"

UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "static/output_videos"
GRAPH_FOLDER = "static/output_graphs"

MAX_FRAMES = 200
NUM_JOINTS = 33
CHANNELS = 3

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(GRAPH_FOLDER, exist_ok=True)

# ======================
# MEDIAPIPE POSE
# ======================
mp_pose = mp.solutions.pose
pose_detector = mp_pose.Pose(static_image_mode=False,
                             model_complexity=1,
                             min_detection_confidence=0.5,
                             min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# ======================
# POSE GRAPH
# ======================
POSE_EDGES = [
    (11, 13), (13, 15),
    (12, 14), (14, 16),
    (23, 25), (25, 27),
    (24, 26), (26, 28),
    (11, 12),
    (23, 24)
]

def build_adjacency(num_nodes=33, edges=POSE_EDGES):
    A = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    for i, j in edges:
        A[i, j] = 1
        A[j, i] = 1
    np.fill_diagonal(A, 1)
    return torch.tensor(A, dtype=torch.float32)

# ======================
# ST-GCN NETWORK
# ======================
class STGCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, A):
        super().__init__()
        self.A = A
        self.gcn = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.tcn = nn.Conv2d(out_channels, out_channels, kernel_size=(9, 1), padding=(4, 0))
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = torch.einsum("nctv,vw->nctw", x, self.A.to(x.device))
        x = self.relu(self.gcn(x))
        x = self.bn(self.tcn(x))
        return self.relu(x)

class RealSTGCN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        A = build_adjacency(NUM_JOINTS, POSE_EDGES)
        self.block1 = STGCNBlock(3, 64, A)
        self.block2 = STGCNBlock(64, 64, A)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = x.mean(dim=[2, 3])
        x = self.dropout(x)
        return self.fc(x)

# ======================
# LOAD MODEL
# ======================
device = "cuda" if torch.cuda.is_available() else "cpu"

class_names = list(np.load(CLASSES_NPY))
num_classes = len(class_names)

model = RealSTGCN(num_classes=num_classes)

try:
    model.load_state_dict(torch.load(MODEL_WEIGHTS, map_location=device))
    model.to(device)
    model.eval()
    print("✔ Model Loaded Successfully")
except Exception as e:
    print("❌ Model Load Error:", e)
    model = None

# ======================
# DATA EXTRACTION
# ======================
def extract_landmarks(video_path):
    cap = cv2.VideoCapture(video_path)
    frames, seq = [], []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frames.append(frame)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = pose_detector.process(rgb)

        if res.pose_landmarks:
            lm = res.pose_landmarks.landmark
            coords = np.array([[lm[i].x, lm[i].y, lm[i].z] for i in range(NUM_JOINTS)])
        else:
            coords = np.zeros((NUM_JOINTS, 3))

        seq.append(coords)

    cap.release()
    return frames, (np.array(seq) if seq else None)

def prepare_input(seq):
    T = seq.shape[0]
    if T < MAX_FRAMES:
        pad = np.zeros((MAX_FRAMES - T, NUM_JOINTS, 3))
        seq = np.vstack((seq, pad))
    else:
        seq = seq[:MAX_FRAMES]

    data = seq.transpose(2, 0, 1)
    return torch.tensor(data).unsqueeze(0).float().to(device)

# ======================
# GRAPH GENERATION
# ======================
def save_movement_graph(landmarks, video_base):
    left_y = landmarks[:, 27, 1]
    right_y = landmarks[:, 28, 1]

    graph_filename = f"{video_base}.png"
    graph_path = os.path.join(GRAPH_FOLDER, graph_filename)

    plt.figure(figsize=(10, 4))
    plt.plot(left_y, label="Left Ankle Y", linewidth=2)
    plt.plot(right_y, label="Right Ankle Y", linewidth=2)
    plt.legend()
    plt.title("Ankle Vertical Movement")
    plt.xlabel("Frame")
    plt.ylabel("Y Position")
    plt.savefig(graph_path)
    plt.close()

    return graph_filename

# ======================
# MAIN PREDICTION
# ======================
def process_and_predict(video_path):
    frames, landmarks = extract_landmarks(video_path)

    if landmarks is None:
        return None, None, None, "No person detected."

    model_input = prepare_input(landmarks)

    with torch.no_grad():
        logits = model(model_input)
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
        cid = int(np.argmax(probs))
        pred = class_names[cid]

    # filenames
    video_base = os.path.splitext(os.path.basename(video_path))[0]
    output_video = f"{video_base}_output.mp4"
    graph_filename = save_movement_graph(landmarks, video_base)

    # draw skeletons
    out_frames = []
    for f in frames:
        rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
        res = pose_detector.process(rgb)
        if res.pose_landmarks:
            mp_drawing.draw_landmarks(
                f, res.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0,255,0), thickness=2),
                connection_drawing_spec=mp_drawing.DrawingSpec(color=(0,255,0), thickness=2)
            )
        out_frames.append(f)

    outpath = os.path.join(OUTPUT_FOLDER, output_video)
    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(outpath, cv2.VideoWriter_fourcc(*"mp4v"), 25, (w, h))
    for f in out_frames:
        writer.write(f)
    writer.release()

    return pred, output_video, graph_filename, None

# ======================
# FLASK
# ======================
app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files["file"]
        if file.filename == "":
            return render_template("index.html", error="No file uploaded")

        save_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(save_path)

        pred, video_out, graph_path, error = process_and_predict(save_path)

        return render_template("index.html",
                               prediction=pred,
                               video_path=video_out,
                               graph_path=graph_path,
                               error=error)

    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
