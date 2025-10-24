import argparse
import pickle as pkl

import numpy as np
from scipy.spatial.transform import Rotation, Slerp


SAFE_Z_HEIGHT = 0.76


def ease_out_cubic(t: float) -> float:
    """Cubic ease-out curve for smooth deceleration."""
    return 1 - (1 - t) ** 3


def ease_in_cubic(t: float) -> float:
    """Cubic ease-in curve for smooth acceleration."""
    return t**3


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert robot motion pickle to CSV format"
    )
    parser.add_argument("input_pkl", help="Input pickle file")
    parser.add_argument("output_csv", help="Output CSV file")
    parser.add_argument(
        "--add-start-transition",
        action="store_true",
        default=True,
        help="Add start transition (default: True)",
    )
    parser.add_argument(
        "--no-start-transition",
        dest="add_start_transition",
        action="store_false",
        help="Don't add start transition",
    )
    parser.add_argument(
        "--add-end-transition",
        action="store_true",
        default=True,
        help="Add end transition (default: True)",
    )
    parser.add_argument(
        "--no-end-transition",
        dest="add_end_transition",
        action="store_false",
        help="Don't add end transition",
    )
    parser.add_argument(
        "--transition-duration",
        type=float,
        default=0.5,
        help="Transition duration in seconds (default: 0.5)",
    )
    parser.add_argument(
        "--pad-duration",
        type=float,
        default=1.0,
        help="Pad duration in seconds (default: 1.0)",
    )
    return parser.parse_args()


def _load_motion(path: str) -> dict:
    with open(path, "rb") as fh:
        return pkl.load(fh)


def _ensure_quaternions(root_rot: np.ndarray) -> tuple[np.ndarray, str]:
    """Return quaternions in XYZW order and report detected layout."""
    if root_rot.shape[1] == 4:
        quats_candidate = np.asarray(root_rot, dtype=np.float64)
        wxyz_mean = np.abs(quats_candidate[:, 0]).mean()
        xyzw_mean = np.abs(quats_candidate[:, 3]).mean()
        if xyzw_mean >= wxyz_mean:
            order = "XYZW"
            quats = quats_candidate
        else:
            order = "WXYZ"
            quats = quats_candidate[:, [1, 2, 3, 0]]
        return quats, order
    if root_rot.shape[1] == 3:
        order = "axis-angle"
        quats = []
        for rot_vec in root_rot:
            angle = np.linalg.norm(rot_vec)
            if angle > 1e-6:
                rotation = Rotation.from_rotvec(rot_vec)
            else:
                rotation = Rotation.from_quat([0, 0, 0, 1])
            quats.append(rotation.as_quat())
        return np.asarray(quats, dtype=np.float64), order
    raise ValueError(f"Unexpected root_rot shape: {root_rot.shape}")


def _apply_yaw_correction(
    root_pos: np.ndarray, quats: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Rotate root data to match MJLab's coordinate frame."""
    correction_rot = Rotation.from_euler("Z", 90, degrees=True)
    corrected_rot = correction_rot * Rotation.from_quat(quats)
    corrected_quats = corrected_rot.as_quat()
    corrected_pos = correction_rot.apply(root_pos)
    return corrected_pos, corrected_quats


def _split_frame(frame: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pos = frame[:3]
    quat = frame[3:7]
    dof = frame[7:]
    return pos, quat, dof


def _build_safe_frame(frame: np.ndarray) -> np.ndarray:
    """Return a safe standing frame aligned with the frame's yaw."""
    safe_pos, target_quat, _ = _split_frame(frame)
    safe_pos = safe_pos.copy()
    safe_pos[2] = max(safe_pos[2], SAFE_Z_HEIGHT)

    yaw = Rotation.from_quat(target_quat).as_euler("ZYX", degrees=False)[0]
    safe_rot = Rotation.from_euler("ZYX", [yaw, 0.0, 0.0], degrees=False)
    safe_quat = safe_rot.as_quat()

    safe_dof = np.zeros_like(frame[7:])
    return np.concatenate([safe_pos, safe_quat, safe_dof])


def _create_transition(
    start_frame: np.ndarray,
    end_frame: np.ndarray,
    num_frames: int,
    ease_fn,
) -> np.ndarray:
    if num_frames <= 0:
        return np.zeros((0, start_frame.shape[0]), dtype=start_frame.dtype)
    if num_frames == 1:
        return start_frame[np.newaxis, :]

    key_rots = Rotation.from_quat(
        np.vstack([start_frame[3:7], end_frame[3:7]])
    )
    slerp = Slerp([0.0, 1.0], key_rots)

    frames = np.zeros((num_frames, start_frame.shape[0]), dtype=start_frame.dtype)
    for i in range(num_frames):
        if num_frames == 1:
            t = 1.0
        else:
            t = i / (num_frames - 1)
        t_eased = ease_fn(t)
        frames[i, :3] = (
            start_frame[:3] * (1 - t_eased) + end_frame[:3] * t_eased
        )
        frames[i, 3:7] = slerp([t_eased]).as_quat()[0]
        frames[i, 7:] = (
            start_frame[7:] * (1 - t_eased) + end_frame[7:] * t_eased
        )
    return frames


def main() -> None:
    args = _parse_args()

    motion_data = _load_motion(args.input_pkl)

    fps = float(motion_data["fps"])
    root_pos = np.asarray(motion_data["root_pos"], dtype=np.float64)
    root_rot = np.asarray(motion_data["root_rot"], dtype=np.float64)
    dof_pos = np.asarray(motion_data["dof_pos"], dtype=np.float64)

    print(f"fps: {fps}")
    print(f"root_pos shape: {root_pos.shape}")
    print(f"root_rot shape: {root_rot.shape}")
    print(f"dof_pos shape: {dof_pos.shape}")

    quats, detected_order = _ensure_quaternions(root_rot)
    print(f"Interpreting root_rot as {detected_order}")

    root_pos, quats = _apply_yaw_correction(root_pos, quats)
    print("Applied +90° yaw correction to orientations and positions.")

    frames = np.concatenate([root_pos, quats, dof_pos], axis=1)

    transition_frames = 0
    if args.transition_duration > 0:
        transition_frames = max(
            2, int(round(args.transition_duration * fps))
        )

    if args.add_start_transition and transition_frames:
        print(
            f"Adding start transition: {args.transition_duration}s "
            f"({transition_frames} frames)"
        )
        safe_start = _build_safe_frame(frames[0])
        start_transition = _create_transition(
            safe_start, frames[0], transition_frames, ease_in_cubic
        )
        frames = np.vstack([start_transition, frames])
    else:
        print("Skipping start transition.")

    if args.add_end_transition and transition_frames:
        print(
            f"Adding end transition: {args.transition_duration}s "
            f"({transition_frames} frames)"
        )
        safe_end = _build_safe_frame(frames[-1])
        end_transition = _create_transition(
            frames[-1], safe_end, transition_frames, ease_out_cubic
        )
        frames = np.vstack([frames, end_transition])
    else:
        print("Skipping end transition.")

    if args.pad_duration > 0:
        pad_frames = int(round(args.pad_duration * fps))
        if pad_frames > 0:
            print(f"Adding pad duration: {args.pad_duration}s ({pad_frames} frames)")
            padding = np.repeat(frames[-1:], pad_frames, axis=0)
            frames = np.vstack([frames, padding])
    else:
        print("No pad duration requested.")

    np.savetxt(args.output_csv, frames, delimiter=",", fmt="%.8f")
    print(f"Saved to {args.output_csv}")
    print(
        f"Final motion: {frames.shape[0]} frames, "
        f"{frames.shape[1]} columns (3 pos, 4 quat, {frames.shape[1] - 7} dof)"
    )


if __name__ == "__main__":
    main()
