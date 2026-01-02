from abc import ABC, abstractmethod
from enum import Enum
from typing import Callable, Generic, Optional, TypeVar, Union

import mediapipe as mp

from .hands_results import HandLandmarkerResult, HandLandmarkerResultProtocol
from ..types import ImageArray, TimestampMs
from ..landmarker import Landmarker

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode


class MediapipeHandLandmarkerRunningMode(Enum):
    """
    Defines the running mode for the synchronous hand landmarker.

    Attributes:
        IMAGE: Single image processing.
        VIDEO: Continuous video processing with timestamp support.
    """

    IMAGE = VisionRunningMode.IMAGE
    VIDEO = VisionRunningMode.VIDEO


class SyncMediapipeHandLandmarker(Landmarker):
    """
    Synchronous MediaPipe Hand Landmarker.

    Processes single images or video frames synchronously and returns
    a HandLandmarkerResult containing landmarks, handedness, and helper methods.

    Args:
        model_path (str): Path to the hand_landmarker.task model file.
        running_mode (MediapipeHandLandmarkerRunningMode): IMAGE or VIDEO mode.
        num_hands (int): Maximum number of hands to detect.
        min_detection_confidence (float): Minimum confidence for detection.
        min_presence_confidence (float): Minimum confidence for hand presence.
        min_tracking_confidence (float): Minimum confidence for tracking.

    Example:
        >>> detector = MediapipeHandLandmarker("hand_landmarker.task",
           running_mode=MediapipeHandLandmarkerRunningMode.IMAGE)
        >>> result = detector.detect(frame)
        >>> image_with_landmarks = result.draw(frame)
    """

    def __init__(
        self,
        model_path: str,
        running_mode: MediapipeHandLandmarkerRunningMode = MediapipeHandLandmarkerRunningMode.IMAGE,
        num_hands: int = 1,
        min_detection_confidence: float = 0.6,
        min_presence_confidence: float = 0.6,
        min_tracking_confidence: float = 0.6,
    ) -> None:
        self._running_mode = running_mode
        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=running_mode.value,
            num_hands=num_hands,
            min_hand_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=min_presence_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._landmarker = HandLandmarker.create_from_options(options)

    def detect_raw(
        self, frame: ImageArray, timestamp_ms: Optional[TimestampMs] = None
    ) -> HandLandmarkerResultProtocol:
        """
        Detect hand landmarks on a single frame without wrapping results.

        Args:
            frame (ImageArray): RGB image as numpy array.
            timestamp_ms (Optional[TimestampMs]): Required for VIDEO mode.

        Returns:
            HandLandmarkerResultProtocol: Raw MediaPipe result.

        Raises:
            RuntimeError: If the detector has been closed.
            ValueError: If timestamp_ms is missing in VIDEO mode.
        """
        if self._landmarker is None:
            raise RuntimeError("Landmarker is closed")
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        match self._running_mode:
            case MediapipeHandLandmarkerRunningMode.IMAGE:
                return self._landmarker.detect(mp_image)
            case MediapipeHandLandmarkerRunningMode.VIDEO:
                if timestamp_ms is None:
                    raise ValueError("timestamp_ms required for VIDEO mode")
                return self._landmarker.detect_for_video(mp_image, timestamp_ms)

    def detect(
        self, frame: ImageArray, timestamp_ms: Optional[TimestampMs] = None
    ) -> HandLandmarkerResult:
        """
        Detect hand landmarks and return a HandLandmarkerResult wrapper.

        Args:
            frame (ImageArray): RGB image as numpy array.
            timestamp_ms (Optional[TimestampMs]): Required for VIDEO mode.

        Returns:
            HandLandmarkerResult: Wrapped result with helper methods.
        """
        return HandLandmarkerResult(self.detect_raw(frame, timestamp_ms))

    def close(self):
        """
        Close the detector and release resources.
        """
        if self._landmarker:
            self._landmarker.close()
            self._landmarker = None


RawCallback = Callable[[HandLandmarkerResultProtocol, mp.Image, TimestampMs], None]
WrappedCallback = Callable[[HandLandmarkerResult, mp.Image, TimestampMs], None]

CallbackType = TypeVar("CallbackType", WrappedCallback, RawCallback)


class AbstractStreamMediapipeHandLandmarker(Landmarker, ABC, Generic[CallbackType]):
    """
    Asynchronous MediaPipe Hand Landmarker for LIVE_STREAM mode.

    Sends frames to MediaPipe asynchronously. Results are delivered
    via a callback function provided at initialization.

    Args:
        model_path (str): Path to the hand_landmarker.task model file.
        result_callback (Callable): Function called on each processed frame.
        Signature: (HandLandmarkerResult, mp.Image, TimestampMs) -> None
        num_hands (int): Maximum number of hands to detect.
        min_detection_confidence (float): Minimum confidence for detection.
        min_presence_confidence (float): Minimum confidence for hand presence.
        min_tracking_confidence (float): Minimum confidence for tracking.

    Example:
        >>> def callback(result, image, ts):
        ...     print("Detected hands:", len(result.hands))
        >>> detector = MediapipeHandLandmarkerStream("hand_landmarker.task", callback)
        >>> detector.send(frame, timestamp_ms)
    """

    def __init__(
        self,
        model_path: str,
        result_callback: CallbackType,
        num_hands: int = 1,
        min_detection_confidence: float = 0.6,
        min_presence_confidence: float = 0.6,
        min_tracking_confidence: float = 0.6,
    ) -> None:
        if result_callback is None:
            raise ValueError("result_callback is required for LIVE_STREAM mode")
        self._result_callback = result_callback
        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=VisionRunningMode.LIVE_STREAM,
            num_hands=num_hands,
            min_hand_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=min_presence_confidence,
            min_tracking_confidence=min_tracking_confidence,
            result_callback=self._internal_callback,
        )

        self._landmarker = HandLandmarker.create_from_options(options)

    @abstractmethod
    def _internal_callback(
        self,
        raw_result: HandLandmarkerResultProtocol,
        mp_image: mp.Image,
        timestamp_ms: int,
    ):
        pass

    def send(self, frame: ImageArray, timestamp_ms: TimestampMs) -> None:
        """
        Send a frame for asynchronous processing.

        Args:
            frame (ImageArray): RGB image as numpy array.
            timestamp_ms (TimestampMs): Timestamp in milliseconds.

        Raises:
            RuntimeError: If the detector has been closed.
        """
        if self._landmarker is None:
            raise RuntimeError("Landmarker is closed")
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        self._landmarker.detect_async(mp_image, timestamp_ms)

    def close(self) -> None:
        """
        Close the detector and release resources.
        """
        if self._landmarker:
            self._landmarker.close()
            self._landmarker = None


class RawStreamMediapipeLandmarker(AbstractStreamMediapipeHandLandmarker[RawCallback]):
    def __init__(
        self,
        model_path: str,
        result_callback: RawCallback,
        num_hands: int = 1,
        min_detection_confidence: float = 0.6,
        min_presence_confidence: float = 0.6,
        min_tracking_confidence: float = 0.6,
    ) -> None:
        super().__init__(
            model_path,
            result_callback,
            num_hands,
            min_detection_confidence,
            min_presence_confidence,
            min_tracking_confidence,
        )

    def _internal_callback(
        self,
        raw_result: HandLandmarkerResultProtocol,
        mp_image: mp.Image,
        timestamp_ms: int,
    ):
        self._result_callback(raw_result, mp_image, timestamp_ms)


class WrappedStreamMediapipeLandmarker(
    AbstractStreamMediapipeHandLandmarker[WrappedCallback]
):
    def __init__(
        self,
        model_path: str,
        result_callback: WrappedCallback,
        num_hands: int = 1,
        min_detection_confidence: float = 0.6,
        min_presence_confidence: float = 0.6,
        min_tracking_confidence: float = 0.6,
    ) -> None:
        super().__init__(
            model_path,
            result_callback,
            num_hands,
            min_detection_confidence,
            min_presence_confidence,
            min_tracking_confidence,
        )

    def _internal_callback(
        self,
        raw_result: HandLandmarkerResultProtocol,
        mp_image: mp.Image,
        timestamp_ms: int,
    ):
        wrapped = HandLandmarkerResult(raw_result)
        self._result_callback(wrapped, mp_image, timestamp_ms)
