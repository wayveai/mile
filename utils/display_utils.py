from typing import Optional
import numpy as np
import tempfile
import os
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
