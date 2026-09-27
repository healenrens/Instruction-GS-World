"""Exact requested native frames with their decoded timestamps."""

from .tracker_visual_review_v67 import decode_case
from .video_file_decoder import read_video_frames, VideoDecodeError


def read_object_video_frames_v69(case, episode_indices):
    record = case["record"]
    if record["adapter"] == "rgb_episode_cache":
        frames = decode_case(case, episode_indices, return_error=True)
        return frames, episode_indices.double()/record["fps"], "rgb_cache_frame_index_over_manifest_fps"
    decoded = read_video_frames(record["path"], episode_indices+record["frame_offset"], record["fps"], return_timestamps=True)
    if isinstance(decoded, VideoDecodeError):
        return decoded, None, "video_pts"
    frames, timestamps = decoded
    return frames.permute(0, 3, 1, 2), timestamps, "video_pts"
