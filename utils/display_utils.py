from typing import Optional
import numpy as np
import tempfile
import os
import PIL
from base64 import b64encode
from gym.wrappers.monitoring.video_recorder import ImageEncoder


def display_video(path):
    """
    A portable way to show video in a notebook
    """
    with open(path, 'rb') as f:
        data_url = "data:video/mp4;base64," + b64encode(f.read()).decode()

    from IPython.display import HTML  # pylint: disable=import-error

    display(  # type: ignore # noqa
        HTML(
            """
          <video controls>
                <source src="%s" type="video/mp4">
          </video>
      """
            % data_url
        )
    )


def make_video_in_temp(debug_frames):
    video_path = tempfile.mkdtemp()
    video_path = os.path.join(video_path, 'temp_video.mp4')

    encoder = ImageEncoder(video_path, debug_frames[0].shape, 25, 25)
    for im in debug_frames:
        encoder.capture_frame(im)
    encoder.close()

    print("Video is saved to ", video_path)
    display_video(video_path)


def overlay_images(
        background_img: np.ndarray, foreground_img: np.ndarray, position, alpha: float = 1.0
) -> np.ndarray:
    background_img = PIL.Image.fromarray(background_img, 'RGB')
    foreground_img = PIL.Image.fromarray(foreground_img, 'RGB')

    a_channel = int(255 * alpha)
    mask = PIL.Image.new('RGBA', foreground_img.size, (0, 0, 0, a_channel))

    background_img.paste(foreground_img, box=position, mask=mask)
    return np.asarray(background_img, dtype=np.uint8)


def downsample(image, scale_factor):
    pil_image = PIL.Image.fromarray(np.uint8(image))
    downsampled_image = pil_image.resize((int(image.shape[1] * scale_factor), int(image.shape[0] * scale_factor)),
                                         resample=PIL.Image.LANCZOS)
    return np.array(downsampled_image)
