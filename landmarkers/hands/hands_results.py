from typing import (
    Literal,
    Optional,
    Protocol,
    Sequence,
    Union,
    cast,
    runtime_checkable,
)

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
    """
    Wrapper around MediaPipe HandLandmarker result.

    Provides helper methods for drawing, extracting features, and
    easy access to landmarks and handedness for one or multiple hands.

    If a hand_index is provided to a method, it returns the data for that hand.
    Otherwise, returns a list/array for all detected hands.
    """

    def __init__(self, data: HandLandmarkerResultProtocol):
        self._data = data

    @property
    def data(self) -> HandLandmarkerResultProtocol | None:
        """Return the raw MediaPipe result."""
        return self._data if self._data else None

    @property
    def hands_count(self) -> int:
        """Return the number of detected hands."""
        if not self.data:
            return 0
        return len(self.data.hand_landmarks)

    def draw(self, image: ImageArray, hand_index: Optional[int] = None) -> ImageArray:
        """
        Draw hand landmarks on a copy of the image.

        Args:
                                                                        image: RGB image as a numpy array.
                                                                        hand_index: Optional index of hand to draw. If None, draws all hands.

        Returns:
                                                                        Image with hand landmarks drawn.
        """
        new_image = image.copy()
        if not self.data:
            return new_image

        hands = (
            [self.data.hand_landmarks[hand_index]]
            if hand_index is not None
            else self.data.hand_landmarks
        )

        for hand in hands:
            for lm in hand:
                h, w, _ = new_image.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(new_image, (cx, cy), 4, (0, 255, 0), -1)

        return new_image

    def landmarks_array(
        self, hand_index: Optional[int] = None
    ) -> Optional[Union[np.ndarray, list[np.ndarray]]]:
        """
        Return landmarks as numpy array(s) of shape (21,3) in image-relative coordinates.

        Args:
                                        hand_index: Optional index of hand. If None, returns a list of arrays for all hands.

        Returns:
                                        np.ndarray or list of np.ndarray. Returns None if no hand detected.
        """
        if not self.data or not self.data.hand_landmarks:
            return None

        hands = (
            [self.data.hand_landmarks[hand_index]]
            if hand_index is not None
            else self.data.hand_landmarks
        )

        result = [
            np.array([[lm.x, lm.y, lm.z] for lm in hand], dtype=np.float32)
            for hand in hands
        ]

        return result[0] if hand_index is not None else result

    def world_landmarks_array(
        self, hand_index: Optional[int] = None
    ) -> Optional[Union[np.ndarray, list[np.ndarray]]]:
        """
        Return hand landmarks as numpy array(s) of shape (21,3) in world coordinates.

        World coordinates are in a 3D metric space, unlike image-relative coordinates
        which are normalized to the image dimensions. Useful for 3D hand pose estimation
        and spatial reasoning.

        Args:
                                        hand_index: Optional index of hand. If None, returns a list of arrays for all hands.

        Returns:
                                        np.ndarray or list of np.ndarray. Returns None if no hand or world landmarks detected.
        """
        if not self.data or not self.data.hand_world_landmarks:
            return None

        hands = (
            [self.data.hand_world_landmarks[hand_index]]
            if hand_index is not None
            else self.data.hand_world_landmarks
        )

        result = [
            np.array([[lm.x, lm.y, lm.z] for lm in hand], dtype=np.float32)
            for hand in hands
        ]

        return result[0] if hand_index is not None else result

    def handedness(
        self, hand_index: Optional[int] = None
    ) -> Optional[
        Union[
            tuple[int, float, str], list[tuple[int, float, str]]
        ]
    ]:
        """
        Return the full handedness category for each hand.

        The returned tuple contains:
                        (index, score, category_name), where category_name is "Left" or "Right".

        Args:
                        hand_index: Optional index of hand. If None, returns a list of tuples for all hands.

        Returns:
                        tuple or list of tuples: Each tuple is (index, score, category_name).
                        Returns None if no hand or handedness data is available.
        """
        if not self.data or not self.data.handedness:
            return None

        hands = (
            [self.data.handedness[hand_index][0]]
            if hand_index is not None
            else [hand[0] for hand in self.data.handedness if hand]
        )

        result = [
            (hand.index, hand.score, hand.category_name)
            for hand in hands
        ]

        return result[0] if hand_index is not None else result
