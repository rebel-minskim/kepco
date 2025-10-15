"""
Configuration module for the client application.
Contains all configurable parameters and settings.
"""
import os
from dataclasses import dataclass
from typing import Tuple, List


@dataclass
class ModelConfig:
    """Model configuration parameters."""
    name: str = "yolov11_ensemble"  # Changed to use JPEG ensemble model
    input_width: int = 800
    input_height: int = 800
    confidence_threshold: float = 0.20
    iou_threshold: float = 0.50
    max_detections: int = 1024
    draw_confidence: float = 0.20


@dataclass
class ServerConfig:
    """Server configuration parameters."""
    url: str = "localhost:8001"
    timeout: float = None
    verbose: bool = False


@dataclass
class VideoConfig:
    """Video processing configuration."""
    fps: float = 24.0
    max_history: int = 2
    distance_threshold: int = 50
    line_thickness: int = 4
    font_scale: float = 0.6
    font_thickness: int = 2


@dataclass
class PathsConfig:
    """File and directory paths configuration."""
    data_yaml: str = "./data.yaml"
    output_dir: str = "./output"
    media_dir: str = "./media"
    
    def __post_init__(self):
        """Create directories if they don't exist."""
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.media_dir, exist_ok=True)


@dataclass
class ClientConfig:
    """Main client configuration."""
    model: ModelConfig = None
    server: ServerConfig = None
    video: VideoConfig = None
    paths: PathsConfig = None
    
    def __post_init__(self):
        if self.model is None:
            self.model = ModelConfig()
        if self.server is None:
            self.server = ServerConfig()
        if self.video is None:
            self.video = VideoConfig()
        if self.paths is None:
            self.paths = PathsConfig()


# Default configuration instance
DEFAULT_CONFIG = ClientConfig()
