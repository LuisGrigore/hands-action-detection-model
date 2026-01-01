from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Iterable, Optional, Tuple, Union
import numpy as np
import cv2


class LandmarkSet:
	"""
	Array-first landmark container.

	Internal shape:
		data: (N, L, D)
		N = instances
		L = landmarks
		D = dimensions (x, y, z, ...)
	"""

	def __init__(
		self,
		data: np.ndarray,
		*,
		image_size: Optional[Tuple[int, int]] = None,
		normalized: bool = True,
		landmark_names: Optional[Iterable[str]] = None,
	):
		if data.ndim != 3:
			raise ValueError("data must have shape (N, L, D)")

		self._data = data.astype(np.float32, copy=False)
		self.image_size = image_size
		self.normalized = normalized

		if landmark_names is not None:
			self._name_to_idx = {name: i for i, name in enumerate(landmark_names)}
		else:
			self._name_to_idx = None

	# ---------- core data ----------

	@property
	def data(self) -> np.ndarray:
		return self._data

	@property
	def num_instances(self) -> int:
		return self._data.shape[0]

	@property
	def num_landmarks(self) -> int:
		return self._data.shape[1]

	@property
	def dim(self) -> int:
		return self._data.shape[2]

	# ---------- semantic access ----------

	def landmark_index(self, name: str) -> int:
		if self._name_to_idx is None:
			raise RuntimeError("landmark names not defined")
		return self._name_to_idx[name]

	def get(self, landmark: Union[int, str]) -> np.ndarray:
		idx = landmark if isinstance(landmark, int) else self.landmark_index(landmark)
		return self._data[:, idx]

	# ---------- normalization ----------

	def normalize(self) -> "LandmarkSet":
		if self.normalized:
			return self

		if self.image_size is None:
			raise RuntimeError("image_size required")

		w, h = self.image_size
		data = self._data.copy()
		data[..., 0] /= w
		data[..., 1] /= h

		return LandmarkSet(
			data,
			image_size=self.image_size,
			normalized=True,
			landmark_names=self._name_to_idx.keys() if self._name_to_idx else None,
		)
		
	def relative_to(self, landmark: Union[int, str]) -> "LandmarkSet":
		"""
		Returns a LandmarkSet where all landmarks are expressed
		relative to the given landmark.
		"""
		idx = landmark if isinstance(landmark, int) else self.landmark_index(landmark)

		data = self._data.copy()
		origin = data[:, idx:idx+1, :]
		data -= origin

		return LandmarkSet(
			data,
			image_size=self.image_size,
			normalized=self.normalized,
			landmark_names=self._name_to_idx.keys() if self._name_to_idx else None,
		)

	def denormalize(self) -> "LandmarkSet":
		if not self.normalized:
			return self

		if self.image_size is None:
			raise RuntimeError("image_size required")

		w, h = self.image_size
		data = self._data.copy()
		data[..., 0] *= w
		data[..., 1] *= h

		return LandmarkSet(
			data,
			image_size=self.image_size,
			normalized=False,
			landmark_names=self._name_to_idx.keys() if self._name_to_idx else None,
		)

	# ---------- geometry ----------

	def bounding_boxes(self) -> np.ndarray:
		xy = self._data[..., :2]
		return np.concatenate([xy.min(axis=1), xy.max(axis=1)], axis=1)

	def centers(self) -> np.ndarray:
		return self._data[..., :2].mean(axis=1)

	# ---------- slicing ----------

	def select_instances(self, indices: Iterable[int]) -> "LandmarkSet":
		data = self._data[np.array(list(indices))]
		return LandmarkSet(
			data,
			image_size=self.image_size,
			normalized=self.normalized,
			landmark_names=self._name_to_idx.keys() if self._name_to_idx else None,
		)

	def select_landmarks(self, indices: Iterable[int]) -> "LandmarkSet":
		idx = np.array(list(indices))
		data = self._data[:, idx]

		names = None
		if self._name_to_idx is not None:
			names = [k for k, i in self._name_to_idx.items() if i in idx]

		return LandmarkSet(
			data,
			image_size=self.image_size,
			normalized=self.normalized,
			landmark_names=names,
		)

	# ---------- drawing ----------

	def draw(
		self,
		image: np.ndarray,
		*,
		connections: Optional[Iterable[Tuple[int, int]]] = None,
		landmark_color: Tuple[int, int, int] = (0, 255, 0),
		connection_color: Tuple[int, int, int] = (255, 0, 0),
		radius: int = 3,
		thickness: int = 2,
		copy: bool = True,
	) -> np.ndarray:
		"""
		Draw landmarks on an image.

		Args:
			image: BGR image
			connections: iterable of (i, j) landmark indices
			copy: if True returns a copy, else draws in-place
		"""
		if copy:
			image = image.copy()

		h, w = image.shape[:2]

		if self.normalized:
			xs = (self._data[..., 0] * w).astype(int)
			ys = (self._data[..., 1] * h).astype(int)
		else:
			xs = self._data[..., 0].astype(int)
			ys = self._data[..., 1].astype(int)

		for n in range(self.num_instances):
			for i in range(self.num_landmarks):
				cv2.circle(
					image,
					(xs[n, i], ys[n, i]),
					radius,
					landmark_color,
					-1,
				)

			if connections is not None:
				for i, j in connections:
					cv2.line(
						image,
						(xs[n, i], ys[n, i]),
						(xs[n, j], ys[n, j]),
						connection_color,
						thickness,
					)

		return image

	# ---------- serialization ----------

	def to_dict(self) -> dict:
		return {
			"data": self._data.tolist(),
			"image_size": self.image_size,
			"normalized": self.normalized,
			"landmark_names": list(self._name_to_idx.keys()) if self._name_to_idx else None,
		}