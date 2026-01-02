from abc import ABC, abstractmethod
from contextlib import AbstractContextManager


class Landmarker(AbstractContextManager, ABC):
	"""
	Abstract base class for a generic landmarker.

	This class provides a standard interface for landmark detection
	classes, including context manager support for automatic resource
	management.

	Methods:
			close(): Release resources used by the landmarker.
			__enter__(): Enter the runtime context related to this object.
			__exit__(): Exit the runtime context and release resources.

	Usage with context manager:
			>>> with SomeLandmarkerImplementation(...) as detector:
			...     detector.detect(frame)
			# Resources are automatically released when exiting the block.
	"""

	@abstractmethod
	def close(self):
		"""
		Close the landmarker and release all associated resources.

		Subclasses must implement this method to properly clean up
		resources such as model memory, threads, or GPU allocations.

		Raises:
				Any exception relevant to resource release failure.
		"""
		pass

	def __enter__(self):
		"""
		Enter the runtime context related to this object.

		Returns:
				self: The landmarker instance for use within the `with` block.
		"""
		return self

	def __exit__(self, exc_type, exc_val, exc_tb):
		"""
		Exit the runtime context and automatically close the landmarker.

		Parameters:
				exc_type (Optional[Type[BaseException]]): Exception type if raised, else None.
				exc_val (Optional[BaseException]): Exception instance if raised, else None.
				exc_tb (Optional[TracebackType]): Traceback object if an exception was raised, else None.

		Returns:
				None
		"""
		self.close()
