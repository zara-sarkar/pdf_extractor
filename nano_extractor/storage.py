"""Storage backend abstractions for saving extracted image crops."""

from abc import ABC, abstractmethod
from io import BytesIO
from pathlib import Path
from PIL import Image


class BaseStorageProvider(ABC):
    """Abstract interface for image crop persistence backends."""

    @abstractmethod
    def save_image(self, image: Image.Image, relative_path: str) -> str:
        """Saves a PIL image and returns a URI/path suitable for Markdown links.

        Args:
            image: PIL Image object to persist.
            relative_path: Key or filename identifier (e.g., 'crops/doc_p1_fig1.png').

        Returns:
            Accessible string URI or file path (e.g., local path or S3 URL).
        """
        pass


class LocalStorageProvider(BaseStorageProvider):
    """Saves image crops directly to the local filesystem."""

    def __init__(self, base_dir: str = "test_results"):
        self.base_dir = Path(base_dir)

    def save_image(self, image: Image.Image, relative_path: str) -> str:
        full_path = self.base_dir / relative_path
        full_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(full_path, format="PNG", optimize=True)
        return str(full_path)


class S3StorageProvider(BaseStorageProvider):
    """Saves image crops to an AWS S3 bucket using boto3."""

    def __init__(self, bucket_name: str, region_name: str = "us-east-1", public_prefix: str = None):
        import boto3
        self.bucket_name = bucket_name
        self.s3_client = boto3.client("s3", region_name=region_name)
        self.public_prefix = public_prefix or f"https://{bucket_name}.s3.{region_name}.amazonaws.com"

    def save_image(self, image: Image.Image, relative_path: str) -> str:
        buffer = BytesIO()
        image.save(buffer, format="PNG", optimize=True)
        buffer.seek(0)

        # Upload binary stream directly to S3 without local disk writes
        self.s3_client.upload_fileobj(
            buffer,
            self.bucket_name,
            relative_path,
            ExtraArgs={"ContentType": "image/png"}
        )
        return f"{self.public_prefix}/{relative_path}"