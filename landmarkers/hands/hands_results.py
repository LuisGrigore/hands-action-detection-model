from copy import deepcopy
from functools import cache
from typing import Literal, Optional, Protocol, Sequence, runtime_checkable

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


def cached_method(method):
    attr_name = f"_cached_{method.__name__}"

    def wrapper(self, *args, **kwargs):
        if not hasattr(self, attr_name):
            bound_method = lambda *a, **kw: method(self, *a, **kw)
            setattr(self, attr_name, cache(bound_method))
        return getattr(self, attr_name)(*args, **kwargs)

    return wrapper


class HandLandmarkerResult:

    def __init__(
        self,
        data: HandLandmarkerResultProtocol,
        num_landmarks: int,
        num_world_landmarks: int,
    ) -> None:

        self._data: HandLandmarkerResultProtocol = deepcopy(data)
        self._num_landmarks: int = num_landmarks
        self._num_world_landmarks: int = num_world_landmarks

    @property
    def data(self) -> HandLandmarkerResultProtocol:
        return deepcopy(self._data)

    @property
    def hands_count(self) -> int:
        return len(self._data.hand_landmarks)

    def draw(self, image: ImageArray, hand_index: Optional[int] = None) -> ImageArray:
        new_image = image.copy()
        hands = self._select_hands(self._data.hand_landmarks, hand_index)
        h, w, _ = new_image.shape
        for hand in hands:
            for lm in hand:
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(new_image, (cx, cy), 4, (0, 255, 0), -1)
        return new_image

    def _pad_or_truncate(self, landmarks: Sequence[Landmark], n: int) -> np.ndarray:
        arr = np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32)
        result = np.zeros((n, 3), dtype=np.float32)
        length = min(len(arr), n)
        result[:length, :] = arr[:length, :]
        return result

    def _select_hands(
        self, hands_source: Sequence[LandmarkList], hand_index: Optional[int]
    ) -> Sequence[LandmarkList]:
        if not hands_source:
            return []
        if hand_index is not None:
            if 0 <= hand_index < len(hands_source):
                return [hands_source[hand_index]]
            return []
        return list(hands_source)

    @cached_method
    def landmarks_array(self, hand_index: Optional[int] = None) -> np.ndarray:
        hands = self._select_hands(self._data.hand_landmarks, hand_index)
        if not hands:
            return np.zeros((0, self._num_landmarks, 3), dtype=np.float32)
        return np.stack(
            [self._pad_or_truncate(hand, self._num_landmarks) for hand in hands], axis=0
        )

    @cached_method
    def world_landmarks_array(self, hand_index: Optional[int] = None) -> np.ndarray:
        hands = self._select_hands(self._data.hand_world_landmarks, hand_index)
        if not hands:
            return np.zeros((0, self._num_world_landmarks, 3), dtype=np.float32)
        return np.stack(
            [self._pad_or_truncate(hand, self._num_world_landmarks) for hand in hands],
            axis=0,
        )

    @cached_method
    def landmarks_array_relative_to_wrist(
        self, hand_index: Optional[int] = None
    ) -> np.ndarray:
        landmarks = self.landmarks_array(hand_index)
        if landmarks.shape[0] == 0:
            return landmarks
        return landmarks - landmarks[:, 0:1, :]

    @cached_method
    def handedness(self, hand_index: Optional[int] = None) -> np.ndarray:
        if not self._data.handedness:
            return np.zeros((0, 3), dtype=np.float32)

        hands_info = [
            hand[0] for hand in self._data.handedness if hand and len(hand) > 0
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
                result = result[hand_index : hand_index + 1, :]
            else:
                result = np.zeros((0, 3), dtype=np.float32)
        return result
