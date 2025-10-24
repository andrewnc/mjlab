"""Convert BVH motion capture files to Unitree G1 CSV trajectories.

This script reads a BVH, retargets the kinematics into the G1 model and writes
a CSV with the exact format expected by mjlabs / MuJoCo controllers and
`pkl_to_csv.py`:  [root_pos(3), quat_xyzw(4), joint_positions(29)].
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Sequence, Tuple, List

import pickle
import numpy as np
import tyro
from scipy.spatial.transform import Rotation

from bvh_io import load_bvh


# ------------------------------- Configuration -------------------------------

# DOF order for the Unitree G1 (29 scalar joints). Do not change without also
# changing consumers downstream.
G1_JOINT_DOF_ORDER: Tuple[str, ...] = (
    # Left leg
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    # Right leg
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    # Waist
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    # Left arm
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    # Right arm
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)

# Constant rotation that maps BVH's Y-up world to MuJoCo's Z-up:
# applying to a *frame* requires conjugation: R_mj = S * R_bvh * S^{-1}
S_BVH_to_MJ = Rotation.from_euler("X", 180.0, degrees=True)


# ------------------------------ Helper functions -----------------------------

def build_joint_channel_lookup(
    joints: Sequence[str],
    channels: Sequence[Sequence[str]],
) -> Dict[str, Tuple[int, Sequence[str]]]:
    """
    Map joint name -> (start index into the motion row, list of channel names).
    """
    lookup: Dict[str, Tuple[int, Sequence[str]]] = {}
    cursor = 0
    for joint, joint_channels in zip(joints, channels):
        lookup[joint] = (cursor, joint_channels)
        cursor += len(joint_channels)
    return lookup


def get_local_rotation(
    frame_row: np.ndarray,
    joint_name: str,
    joint_lookup: Dict[str, Tuple[int, Sequence[str]]],
) -> Rotation:
    """
    Reconstruct the joint-local rotation from the BVH row, honoring the channel order.
    Returns identity() if the joint has no rotation channels.
    """
    start, chs = joint_lookup[joint_name]
    axes: List[str] = []
    angles_deg: List[float] = []
    for local_idx, ch in enumerate(chs):
        if ch.endswith("rotation"):
            axes.append(ch[0].upper())  # 'Xrotation' -> 'X'
            angles_deg.append(frame_row[start + local_idx])
    if not angles_deg:
        return Rotation.identity()
    try:
        return Rotation.from_euler("".join(axes), angles_deg, degrees=True)
    except ValueError as exc:
        raise RuntimeError(f"Failed to build rotation for joint '{joint_name}' "
                           f"from channels {chs}.") from exc


def get_root_translation_mj(
    frame_row: np.ndarray,
    root_joint: str,
    joint_lookup: Dict[str, Tuple[int, Sequence[str]]],
    scale: float,
    order: Tuple[str, str, str],
    signs: Tuple[int, int, int],
) -> np.ndarray:
    """
    Extract the root translation channels and convert BVH (Y-up, likely in cm)
    into MuJoCo (Z-up, meters).
    """
    start, chs = joint_lookup[root_joint]
    raw = {}
    for local_idx, ch in enumerate(chs):
        if ch.endswith("position"):
            axis = ch[0].lower()
            raw[axis] = frame_row[start + local_idx]
    if set(raw.keys()) != {"x", "y", "z"}:
        raise RuntimeError(f"Root '{root_joint}' must have X/Y/Z position; got {sorted(raw.keys())}.")
    # Reorder and apply signs, then scale.
    vec = np.array([raw[order[i]] for i in range(3)], dtype=np.float64)
    vec *= np.array(signs, dtype=np.float64)
    vec *= scale
    return vec


def conjugate_bvh_to_mj(R_bvh: Rotation) -> Rotation:
    """Return S * R_bvh * S^{-1} to express BVH rotation in MuJoCo (Z-up) basis."""
    return S_BVH_to_MJ * R_bvh * S_BVH_to_MJ.inv()


def euler_decompose(R: Rotation, order: str) -> np.ndarray:
    """Convenience wrapper that always returns radians."""
    return R.as_euler(order, degrees=False)


def clamp(val: float, lo: float | None, hi: float | None) -> float:
    if lo is not None and val < lo:
        return lo
    if hi is not None and val > hi:
        return hi
    return val


# ------------------------------- Main converter -------------------------------

# Mechanical axis order of multi-DOF joints in the G1 model:
#  - Hips:    Y (pitch) → X (roll) → Z (yaw)
#  - Ankles:  Y (pitch) → X (roll)
#  - Waist:   Z (yaw) → X (roll) → Y (pitch)
#  - Shoulders: Y (pitch) → X (roll) → Z (yaw)
#  - Wrists:  X (roll) → Y (pitch) → Z (yaw)
#
# These match g1.xml joint axis attributes (axis="0 1 0" etc.) and ranges.

@dataclass
class Args:
    bvh_file: Path
    csv_file: Path
    pkl_file: Path | None = None

    # BVH translation scale (most BVHs are in centimeters → meters).
    translation_scale: float = 0.01

    # Root translation axis mapping BVH→MuJoCo.
    # For proper Y-up→Z-up, the default *must* be ("x","z","y") and (1, 1, -1),
    # i.e., [x, z, -y].
    root_axis_order: Tuple[str, str, str] = ("x", "z", "y")
    root_axis_sign: Tuple[int, int, int] = (1, 1, -1)

    # Optionally clamp to G1 joint limits from the XML. If False, no clamping.
    clamp_to_limits: bool = True


# Joint limits (radians). Pulled from g1.xml; only symmetric bounds are listed
# here for clarity. You can extend per-joint if asymmetric. Values chosen to be
# safely within the XML ranges.
G1_JOINT_LIMITS: Dict[str, Tuple[float | None, float | None]] = {
    # Legs
    "left_hip_pitch_joint": (-2.53,  2.88),
    "left_hip_roll_joint":  (-0.52,  2.97),   # note: g1.xml left hip roll is [-0.52, 2.97]
    "left_hip_yaw_joint":   (-2.76,  2.76),
    "left_knee_joint":      (-0.09,  2.88),
    "left_ankle_pitch_joint": (-0.87, 0.52),
    "left_ankle_roll_joint":  (-0.26, 0.26),
    "right_hip_pitch_joint": (-2.53,  2.88),
    "right_hip_roll_joint":  (-2.97,  0.52),  # mirrored
    "right_hip_yaw_joint":   (-2.76,  2.76),
    "right_knee_joint":      (-0.09,  2.88),
    "right_ankle_pitch_joint": (-0.87, 0.52),
    "right_ankle_roll_joint":  (-0.26, 0.26),
    # Waist
    "waist_yaw_joint":      (-2.618, 2.618),
    "waist_roll_joint":     (-0.52, 0.52),
    "waist_pitch_joint":    (-0.52, 0.52),
    # Arms
    "left_shoulder_pitch_joint": (-3.0892, 2.6704),
    "left_shoulder_roll_joint":  (-1.5882, 2.2515),
    "left_shoulder_yaw_joint":   (-2.618,  2.618),
    "left_elbow_joint":          (-1.0472, 2.0944),
    "left_wrist_roll_joint":     (-1.97222, 1.97222),
    "left_wrist_pitch_joint":    (-1.61443, 1.61443),
    "left_wrist_yaw_joint":      (-1.61443, 1.61443),
    "right_shoulder_pitch_joint": (-3.0892, 2.6704),
    "right_shoulder_roll_joint":  (-2.2515, 1.5882),
    "right_shoulder_yaw_joint":   (-2.618,  2.618),
    "right_elbow_joint":          (-1.0472, 2.0944),
    "right_wrist_roll_joint":     (-1.97222, 1.97222),
    "right_wrist_pitch_joint":    (-1.61443, 1.61443),
    "right_wrist_yaw_joint":      (-1.61443, 1.61443),
}


REQUIRED_BVH_JOINTS: Tuple[str, ...] = (
    "pelvis",
    # Torso chain
    "spine1",
    "spine2",
    "spine3",
    # Legs
    "left_hip",
    "left_knee",
    "left_ankle",
    "right_hip",
    "right_knee",
    "right_ankle",
    # Arms
    "left_shoulder",
    "left_elbow",
    "left_wrist",
    "right_shoulder",
    "right_elbow",
    "right_wrist",
)


class ConversionError(RuntimeError):
    pass


def convert_bvh(args: Args) -> None:
    # Load BVH
    offsets, hierarchy, joints, channels, motion, frames, frame_time = load_bvh(
        str(args.bvh_file), include_motion=True
    )
    if motion is None or frame_time is None:
        raise ConversionError("BVH file has no motion data.")

    # Build lookup and sanity checks
    joint_lookup = build_joint_channel_lookup(joints, channels)
    missing = [j for j in REQUIRED_BVH_JOINTS if j not in joint_lookup]
    if missing:
        raise ConversionError(f"Missing required joints in BVH: {missing}")

    root_joint = joints[0]

    frames_out: List[np.ndarray] = []

    # Process frames
    for t in range(motion.shape[0]):
        row = motion[t]

        # Root translation in MuJoCo basis
        root_pos = get_root_translation_mj(
            row, root_joint, joint_lookup,
            scale=args.translation_scale,
            order=args.root_axis_order,
            signs=args.root_axis_sign,
        )

        # Root orientation (world) in MuJoCo basis
        R_root_bvh = get_local_rotation(row, root_joint, joint_lookup)
        # R_root_mj = conjugate_bvh_to_mj(R_root_bvh)
        R_root_mj = S_BVH_to_MJ * R_root_bvh
        root_rotvec = R_root_mj.as_rotvec()

        # ------------------------ Torso / Waist (3 DOF) -----------------------
        # Relative pelvis->torso rotation in BVH: spine1 * spine2 * spine3
        R_sp1 = get_local_rotation(row, "spine1", joint_lookup)
        R_sp2 = get_local_rotation(row, "spine2", joint_lookup)
        R_sp3 = get_local_rotation(row, "spine3", joint_lookup)
        # Compose in parent->child order (sp1 applied first):
        R_torso_bvh = R_sp3 * R_sp2 * R_sp1
        # Express in MuJoCo basis
        # R_torso_mj = conjugate_bvh_to_mj(R_torso_bvh)
        # Decompose in the mechanical order of waist joints: Z (yaw) → X (roll) → Y (pitch)
        yaw_z, roll_x, pitch_y = euler_decompose(R_torso_bvh, "ZXY")

        joint_vals: Dict[str, float] = {}
        joint_vals["waist_yaw_joint"] = yaw_z
        joint_vals["waist_roll_joint"] = roll_x
        joint_vals["waist_pitch_joint"] = pitch_y

        # ---------------------------- Left leg --------------------------------
        # R_lhip_mj = conjugate_bvh_to_mj(get_local_rotation(row, "left_hip", joint_lookup))
        R_lhip_bvh = get_local_rotation(row, "left_hip", joint_lookup)
        # Decompose in G1 order: Y (pitch), X (roll), Z (yaw)
        lhip_pitch_y, lhip_roll_x, lhip_yaw_z = euler_decompose(R_lhip_bvh, "YXZ")
        joint_vals["left_hip_pitch_joint"] = lhip_pitch_y
        joint_vals["left_hip_roll_joint"]  = lhip_roll_x
        joint_vals["left_hip_yaw_joint"]   = lhip_yaw_z

        R_lknee_bvh = get_local_rotation(row, "left_knee", joint_lookup)
        # Knee is a single Y (pitch) hinge
        lknee_pitch_y = euler_decompose(R_lknee_bvh, "YXZ")[0]
        joint_vals["left_knee_joint"] = lknee_pitch_y

        R_lankle_bvh = get_local_rotation(row, "left_ankle", joint_lookup)
        # Ankle: Y (pitch) → X (roll)
        lankle_pitch_y, lankle_roll_x, _ = euler_decompose(R_lankle_bvh, "YXZ")
        joint_vals["left_ankle_pitch_joint"] = lankle_pitch_y
        joint_vals["left_ankle_roll_joint"]  = lankle_roll_x

        # ---------------------------- Right leg -------------------------------
        R_rhip_bvh = get_local_rotation(row, "right_hip", joint_lookup)
        rhip_pitch_y, rhip_roll_x, rhip_yaw_z = euler_decompose(R_rhip_bvh, "YXZ")
        joint_vals["right_hip_pitch_joint"] = rhip_pitch_y
        joint_vals["right_hip_roll_joint"]  = rhip_roll_x
        joint_vals["right_hip_yaw_joint"]   = rhip_yaw_z

        R_rknee_bvh = get_local_rotation(row, "right_knee", joint_lookup)
        rknee_pitch_y = euler_decompose(R_rknee_bvh, "YXZ")[0]
        joint_vals["right_knee_joint"] = rknee_pitch_y

        R_rankle_bvh = get_local_rotation(row, "right_ankle", joint_lookup)
        rankle_pitch_y, rankle_roll_x, _ = euler_decompose(R_rankle_bvh, "YXZ")
        joint_vals["right_ankle_pitch_joint"] = rankle_pitch_y
        joint_vals["right_ankle_roll_joint"]  = rankle_roll_x

        # ---------------------------- Left arm --------------------------------
        R_lsho_bvh = get_local_rotation(row, "left_shoulder", joint_lookup)
        lsho_pitch_y, lsho_roll_x, lsho_yaw_z = euler_decompose(R_lsho_bvh, "YXZ")
        joint_vals["left_shoulder_pitch_joint"] = lsho_pitch_y
        joint_vals["left_shoulder_roll_joint"]  = lsho_roll_x
        joint_vals["left_shoulder_yaw_joint"]   = lsho_yaw_z

        R_lelb_bvh = get_local_rotation(row, "left_elbow", joint_lookup)
        lelb_pitch_y = euler_decompose(R_lelb_bvh, "YXZ")[0]
        joint_vals["left_elbow_joint"] = lelb_pitch_y

        R_lwri_bvh = get_local_rotation(row, "left_wrist", joint_lookup)
        # Wrist: X (roll) → Y (pitch) → Z (yaw)
        lwri_roll_x, lwri_pitch_y, lwri_yaw_z = euler_decompose(R_lwri_bvh, "XYZ")
        joint_vals["left_wrist_roll_joint"]  = lwri_roll_x
        joint_vals["left_wrist_pitch_joint"] = lwri_pitch_y
        joint_vals["left_wrist_yaw_joint"]   = lwri_yaw_z

        # ---------------------------- Right arm -------------------------------
        R_rsho_bvh = get_local_rotation(row, "right_shoulder", joint_lookup)
        rsho_pitch_y, rsho_roll_x, rsho_yaw_z = euler_decompose(R_rsho_bvh, "YXZ")
        joint_vals["right_shoulder_pitch_joint"] = rsho_pitch_y
        joint_vals["right_shoulder_roll_joint"]  = rsho_roll_x
        joint_vals["right_shoulder_yaw_joint"]   = rsho_yaw_z

        R_relb_bvh = get_local_rotation(row, "right_elbow", joint_lookup)
        relb_pitch_y = euler_decompose(R_relb_bvh, "YXZ")[0]
        joint_vals["right_elbow_joint"] = relb_pitch_y

        R_rwri_bvh = get_local_rotation(row, "right_wrist", joint_lookup)
        rwri_roll_x, rwri_pitch_y, rwri_yaw_z = euler_decompose(R_rwri_bvh, "XYZ")
        joint_vals["right_wrist_roll_joint"]  = rwri_roll_x
        joint_vals["right_wrist_pitch_joint"] = rwri_pitch_y
        joint_vals["right_wrist_yaw_joint"]   = rwri_yaw_z

        # Optional clamping
        if args.clamp_to_limits:
            for name, (lo, hi) in G1_JOINT_LIMITS.items():
                if name in joint_vals:
                    joint_vals[name] = clamp(joint_vals[name], lo, hi)

        # Pack output frame
        joints_vec = np.array([joint_vals[name] for name in G1_JOINT_DOF_ORDER], dtype=np.float64)
        frame_out = np.concatenate([root_pos, root_rotvec, joints_vec], axis=0)
        frames_out.append(frame_out)

    frames_out = np.vstack(frames_out)
    fps = 1.0 / frame_time if frame_time and frame_time > 0 else None

    # Optionally save intermediate MimicKit-style pickle (axis-angle root rot).
    if args.pkl_file is not None:
        args.pkl_file.parent.mkdir(parents=True, exist_ok=True)
        with open(args.pkl_file, "wb") as f:
            pickle.dump(
                {
                    "loop_mode": "wrap",
                    "fps": fps,
                    "frames": frames_out,
                },
                f,
            )

    # Convert root rotvec → XYZW quaternion and write CSV:
    root_pos = frames_out[:, 0:3]
    root_rotvec = frames_out[:, 3:6]
    root_quat_xyzw = Rotation.from_rotvec(root_rotvec).as_quat()  # [x, y, z, w]
    joint_dofs = frames_out[:, 6:]
    csv_data = np.concatenate([root_pos, root_quat_xyzw, joint_dofs], axis=1)

    args.csv_file.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(args.csv_file, csv_data, delimiter=",", fmt="%.8f")

    print(f"Converted {frames_out.shape[0]} frames from {args.bvh_file} → {args.csv_file}")
    if fps is not None:
        print(f"Source FPS: {fps:.3f}")
    if args.pkl_file is not None:
        print(f"Intermediate pickle saved to {args.pkl_file}")


def main(args: Args) -> None:
    convert_bvh(args)


if __name__ == "__main__":
    main(tyro.cli(Args))
