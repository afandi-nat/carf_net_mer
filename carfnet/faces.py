"""Face detection and 68-point facial landmarks.

Detector : YuNet (OpenCV Zoo), falling back to the bundled Haar cascade.
Landmarks: LBF 68 points (`cv2.face`), or dlib when explicitly requested.

Models are downloaded once into `weights/` at the repository root.
"""

import os
import urllib.request

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
WEIGHT_DIR = os.environ.get("CARF_WEIGHT_DIR",
                            os.path.abspath(os.path.join(HERE, "..", "weights")))

URLS = {
    "yunet": "https://github.com/opencv/opencv_zoo/raw/main/models/"
             "face_detection_yunet/face_detection_yunet_2023mar.onnx",
    "lbf": "https://raw.githubusercontent.com/kurnianggoro/GSOC2017/master/data/"
           "lbfmodel.yaml",
    "dlib": "https://github.com/davisking/dlib-models/raw/master/"
            "shape_predictor_68_face_landmarks.dat.bz2",
}
FILES = {
    "yunet": "face_detection_yunet.onnx",
    "lbf": "lbfmodel.yaml",
    "dlib": "shape_predictor_68_face_landmarks.dat",
}


def ensure_model(kind):
    """Return the local path of a model file, downloading it on first use."""
    path = os.path.join(WEIGHT_DIR, FILES[kind])
    if os.path.isfile(path):
        return path
    os.makedirs(WEIGHT_DIR, exist_ok=True)
    print(f">> Downloading the {kind} model (one time only)")
    if kind == "dlib":
        import bz2

        archive = path + ".bz2"
        urllib.request.urlretrieve(URLS[kind], archive)
        with bz2.open(archive, "rb") as src, open(path, "wb") as dst:
            dst.write(src.read())
        os.remove(archive)
    else:
        tmp = path + ".part"
        urllib.request.urlretrieve(URLS[kind], tmp)
        os.replace(tmp, path)
    return path


class FaceDetector:
    """Largest face box. Large frames are downscaled before detection."""

    def __init__(self, kind="auto", max_side=640):
        self.max_side = max_side
        self.kind = None
        if kind in ("auto", "yunet"):
            try:
                self.net = cv2.FaceDetectorYN.create(ensure_model("yunet"), "",
                                                     (320, 320))
                self.kind = "yunet"
            except Exception as exc:  # noqa: BLE001
                if kind == "yunet":
                    raise
                print(f">> YuNet unavailable ({exc}); falling back to Haar")
        if self.kind is None:
            cascade = os.path.join(cv2.data.haarcascades,
                                   "haarcascade_frontalface_default.xml")
            self.haar = cv2.CascadeClassifier(cascade)
            if self.haar.empty():
                raise SystemExit(f"Could not load the Haar cascade at {cascade}")
            self.kind = "haar"

    def __call__(self, img_bgr):
        height, width = img_bgr.shape[:2]
        scale = min(1.0, self.max_side / max(height, width))
        small = cv2.resize(img_bgr, None, fx=scale, fy=scale) if scale < 1 else img_bgr
        if self.kind == "yunet":
            self.net.setInputSize((small.shape[1], small.shape[0]))
            _, faces = self.net.detect(small)
            if faces is None or len(faces) == 0:
                return None
            x, y, w, h = max(faces, key=lambda f: f[2] * f[3])[:4]
        else:
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            faces = self.haar.detectMultiScale(gray, 1.1, 5, minSize=(40, 40))
            if len(faces) == 0:
                return None
            x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        return np.array([x, y, w, h], dtype=np.float32) / scale


class Landmarker:
    """68 iBUG landmarks in full-image coordinates."""

    def __init__(self, kind="lbf"):
        self.kind = kind
        if kind == "lbf":
            if not hasattr(cv2, "face"):
                raise SystemExit(
                    "cv2.face is missing. Install opencv-contrib-python and make "
                    "sure no other opencv package is installed alongside it."
                )
            self.model = cv2.face.createFacemarkLBF()
            self.model.loadModel(ensure_model("lbf"))
        elif kind == "dlib":
            import dlib

            self.dlib = dlib
            self.model = dlib.shape_predictor(ensure_model("dlib"))
        else:
            raise SystemExit(f"Unknown landmark backend: {kind}")

    def __call__(self, img_bgr, box):
        x, y, w, h = [float(v) for v in box]
        if self.kind == "lbf":
            gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            ok, pts = self.model.fit(gray, np.array([[x, y, w, h]], dtype=np.int32))
            if not ok or len(pts) == 0:
                return None
            return np.asarray(pts[0]).reshape(-1, 2).astype(np.float32)
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        rect = self.dlib.rectangle(int(x), int(y), int(x + w), int(y + h))
        shape = self.model(rgb, rect)
        return np.array([(p.x, p.y) for p in shape.parts()], dtype=np.float32)


def plausible(pts, box):
    """Reject obviously wrong fits, e.g. eyes below the mouth."""
    if pts is None or len(pts) != 68 or not np.isfinite(pts).all():
        return False
    eyes_y = pts[36:48, 1].mean()
    mouth_y = pts[48:68, 1].mean()
    iod = np.linalg.norm(pts[36:42].mean(0) - pts[42:48].mean(0))
    return mouth_y > eyes_y and iod > 0.15 * box[2]
