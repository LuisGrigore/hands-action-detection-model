from abc import ABC, abstractmethod
from typing import Callable, Generic, TypeVar

import mediapipe as mp

from .hands_results import HandLandmarkerResult, HandLandmarkerResultProtocol
from ..types import ImageArray, TimestampMs
from ..landmarker import Landmarker

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

RawCallback = Callable[[HandLandmarkerResultProtocol, mp.Image, TimestampMs], None]
WrappedCallback = Callable[[HandLandmarkerResult, mp.Image, TimestampMs], None]

CallbackType = TypeVar("CallbackType", WrappedCallback, RawCallback)


class AbstractStreamMediapipeHandLandmarker(Landmarker, ABC, Generic[CallbackType]):
	"""
	Abstract base class for an asynchronous MediaPipe Hand Landmarker in LIVE_STREAM mode.

	Frames are sent to MediaPipe asynchronously, and results are delivered
	through a user-provided callback function. Subclasses must implement
	`_internal_callback` to define how the raw results are processed or wrapped.

	Parameters:
			model_path (str): Path to the hand_landmarker.task model file.
			result_callback (Callable): Function called for each processed frame.
					Signature depends on the type of callback:
					- RawCallback: (HandLandmarkerResultProtocol, mp.Image, TimestampMs) -> None
					- WrappedCallback: (HandLandmarkerResult, mp.Image, TimestampMs) -> None
			num_hands (int, optional): Maximum number of hands to detect. Defaults to 1.
			min_detection_confidence (float, optional): Minimum confidence for hand detection. Defaults to 0.6.
			min_presence_confidence (float, optional): Minimum confidence for hand presence. Defaults to 0.6.
			min_tracking_confidence (float, optional): Minimum confidence for hand tracking. Defaults to 0.6.

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
		"""
		Internal callback called by MediaPipe when a frame has been processed.

		Subclasses must implement this method to handle the raw result, e.g.,
		passing it directly to the user callback or wrapping it in a result class.

		Parameters:
				raw_result (HandLandmarkerResultProtocol): Raw MediaPipe hand landmark result.
				mp_image (mp.Image): The MediaPipe image corresponding to the frame.
				timestamp_ms (int): Timestamp of the frame in milliseconds.
		"""
		pass

	def send(self, frame: ImageArray, timestamp_ms: TimestampMs) -> None:
		"""
		Send a frame for asynchronous processing.

		Parameters:
				frame (ImageArray): RGB image as a numpy array.
				timestamp_ms (TimestampMs): Timestamp of the frame in milliseconds.

		Raises:
				RuntimeError: If the detector has already been closed.
		"""
		if self._landmarker is None:
			raise RuntimeError("Landmarker is closed")
		mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
		self._landmarker.detect_async(mp_image, timestamp_ms)

	def close(self) -> None:
		"""
		Close the detector and release all resources.
		"""
		if self._landmarker:
			self._landmarker.close()
			self._landmarker = None


class RawStreamMediapipeLandmarker(AbstractStreamMediapipeHandLandmarker[RawCallback]):
	"""
	MediaPipe Hand Landmarker stream using raw MediaPipe results.

	The user callback receives the raw `HandLandmarkerResultProtocol` object
	without any additional wrapping.
	"""

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
		"""Pass the raw MediaPipe result directly to the user callback."""
		self._result_callback(raw_result, mp_image, timestamp_ms)


class WrappedStreamMediapipeLandmarker(
	AbstractStreamMediapipeHandLandmarker[WrappedCallback]
):
	"""
	MediaPipe Hand Landmarker stream using wrapped results.

	The user callback receives a `HandLandmarkerResult` object, which wraps
	the raw MediaPipe result and provides helper methods for easier usage.
	"""

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
		"""
		Wrap the raw MediaPipe result in `HandLandmarkerResult` before
		calling the user callback.
		"""
		wrapped = HandLandmarkerResult(raw_result,21,21)
		self._result_callback(wrapped, mp_image, timestamp_ms)
