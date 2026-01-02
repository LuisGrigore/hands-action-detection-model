from typing import Literal, Optional, Protocol, Sequence, Union, runtime_checkable

import cv2
import numpy as np

from ..types import ImageArray

HandednessLabel = Literal["Left", "Right"]


@runtime_checkable
class Category(Protocol):
	index: int
	score: float
	category_name: HandednessLabel


@runtime_checkable
class Landmark(Protocol):
	x: float
	y: float
	z: float


LandmarkList = Sequence[Landmark]


@runtime_checkable
class HandLandmarkerResultProtocol(Protocol):
	handedness: Sequence[Sequence[Category]]
	hand_landmarks: Sequence[LandmarkList]
	hand_world_landmarks: Sequence[LandmarkList]


class HandLandmarkerResult:
	def __init__(
		self,
		data: HandLandmarkerResultProtocol,
		num_landmarks: int = 21,
		num_world_landmarks: int = 21,
	):
		self._data = data
		self._num_landmarks = num_landmarks
		self._num_world_landmarks = num_world_landmarks

	@property
	def data(self) -> HandLandmarkerResultProtocol:
		return self._data

	@property
	def hands_count(self) -> int:
		return len(self.data.hand_landmarks)

	def draw(self, image: ImageArray, hand_index: Optional[int] = None) -> ImageArray:
		new_image = image.copy()
		hands = self._select_hands(self.data.hand_landmarks, hand_index)

		for hand in hands:
			for lm in hand:
				h, w, _ = new_image.shape
				cx, cy = int(lm.x * w), int(lm.y * h)
				cv2.circle(new_image, (cx, cy), 4, (0, 255, 0), -1)
		return new_image

	def _pad_or_truncate(self, landmarks: Sequence[Landmark], n: int) -> np.ndarray:
		arr = np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32)
		result = np.zeros((n, 3), dtype=np.float32)
		length = min(len(arr), n)
		result[:length, :] = arr[:length, :]
		return result

	def _get_landmarks_array(
		self,
		landmarks_source: Sequence[LandmarkList],
		n_landmarks: int,
		hand_index: Optional[int] = None,
	) -> np.ndarray:
		hands = self._select_hands(landmarks_source, hand_index)
		if not hands:
			return np.zeros((0, n_landmarks, 3), dtype=np.float32)
		return np.stack(
			[self._pad_or_truncate(hand, n_landmarks) for hand in hands], axis=0
		)

	def _select_hands(
		self, hands_source: Sequence[LandmarkList], hand_index: Optional[int]
	) -> Sequence[LandmarkList]:
		if not hands_source:
			return []
		if hand_index is not None:
			if 0 <= hand_index < len(hands_source):
				return [hands_source[hand_index]]
			return []
		return list(hands_source) if isinstance(hands_source, tuple) else hands_source

	def landmarks_array(self, hand_index: Optional[int] = None) -> np.ndarray:
		return self._get_landmarks_array(
			self.data.hand_landmarks, self._num_landmarks, hand_index
		)

	def world_landmarks_array(self, hand_index: Optional[int] = None) -> np.ndarray:
		return self._get_landmarks_array(
			self.data.hand_world_landmarks, self._num_world_landmarks, hand_index
		)

	def landmarks_array_relative_to_wrist(
		self, hand_index: Optional[int] = None
	) -> np.ndarray:
		landmarks = self.landmarks_array(hand_index)
		if landmarks.shape[0] == 0:
			return landmarks
		return landmarks - landmarks[:, 0:1, :]

	def handedness(self, hand_index: Optional[int] = None) -> np.ndarray:
		if not self.data.handedness:
			return np.zeros((0, 3), dtype=np.float32)

		hands_info = [
			hand[0] for hand in self.data.handedness if hand and len(hand) > 0
		]
		result = np.array(
			[
				[hand.index, hand.score, 0 if hand.category_name == "Left" else 1]
				for hand in hands_info
			],
			dtype=np.float32,
		)

		if hand_index is not None:
			if 0 <= hand_index < len(result):
				return result[hand_index : hand_index + 1, :]
			return np.zeros((0, 3), dtype=np.float32)

		return result
