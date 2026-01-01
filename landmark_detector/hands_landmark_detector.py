import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
from typing import Optional
from .landmark_detector import ILandmarkDetector
from .landmark_set import LandmarkSet


class MediaPipeHandDetector(ILandmarkDetector):
    """
    MediaPipe Hands wrapper that outputs LandmarkSet.
    """

    NUM_LANDMARKS = 21
    LANDMARK_DIM = 3  # x, y, z

    def __init__(
        self,
        *,
        task_path: str,
        num_hands: int = 1,
        min_hand_detection_confidence: float = 0.6,
        min_hand_presence_confidence: float = 0.6,
        min_tracking_confidence: float = 0.6,
        running_mode = vision.RunningMode.VIDEO,
        landmark_names: Optional[list[str]] = None,
    ):
        BaseOptions = python.BaseOptions
        HandLandmarkerOptions = vision.HandLandmarkerOptions
        HandLandmarker = vision.HandLandmarker

        self._running_mode = running_mode

        self._options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=task_path),
            running_mode=running_mode,
            num_hands=num_hands,
            min_hand_detection_confidence=min_hand_detection_confidence,
            min_hand_presence_confidence=min_hand_presence_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

        self._landmarker = HandLandmarker.create_from_options(self._options)
        self._landmark_names = landmark_names

    # ------------------------------------------------------------------
    # LandmarkDetector API
    # ------------------------------------------------------------------

    def detect(
        self,
        image: np.ndarray,
        *,
        timestamp_ms: Optional[int] = None,
    ) -> LandmarkSet:
        """
        Detect hands and return a LandmarkSet with:
            data shape: (N, 21, 3)
            coordinates: normalized w.r.t image (MediaPipe native)
        """
        if self._running_mode == vision.RunningMode.VIDEO and timestamp_ms is None:
            raise ValueError("timestamp_ms is required in VIDEO mode")

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=image,
        )

        if self._running_mode == vision.RunningMode.VIDEO:
            result = self._landmarker.detect_for_video(mp_image, timestamp_ms)
        else:
            result = self._landmarker.detect(mp_image)

        if not result.hand_landmarks:
            data = np.zeros(
                (0, self.NUM_LANDMARKS, self.LANDMARK_DIM),
                dtype=np.float32,
            )
        else:
            hands = []
            for hand in result.hand_landmarks:
                coords = np.array(
                    [[lm.x, lm.y, lm.z] for lm in hand],
                    dtype=np.float32,
                )
                hands.append(coords)

            data = np.stack(hands, axis=0)

        h, w = image.shape[:2]

        return LandmarkSet(
            data,
            image_size=(w, h),
            normalized=True,              # normalizado a imagen
            landmark_names=self._landmark_names,
        )

    def reset(self) -> None:
        # MediaPipe no expone reset explícito
        pass

    def close(self) -> None:
        self._landmarker.close()
