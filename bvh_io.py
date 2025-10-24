from __future__ import annotations

import numpy as np

def load_bvh(
    filepath: str,
    include_motion: bool = False,
    include_end_site: bool = False,
    normalize: bool = False,
) -> tuple[
    np.ndarray, list[int], list[str], list[str], np.ndarray | None, int | None, float | None
]:
    """
    Parse a BVH file and return the local offsets, hierarchy, and joints.

    Args:
        filepath: The path to the BVH file.
        include_motion: Whether to include the motion data.
        include_end_site: Whether to include offset information for end sites.
        normalize: Whether to normalize the offsets and motion data to a character height of 1 unit.

    Returns:
        offsets: The local offsets of the joints
        hierarchy: The parent index of each joint
        joints: The names of the joints.
        channels: List of channel types for each joint.
        motion_data: The motion data (if include_motion is True).
        frames: Number of frames (if include_motion is True).
        frame_time: Time per frame (if include_motion is True).
    """
    offsets = []
    hierarchy = []
    joints = []
    channels = []
    stack = []
    current_parent: int = -1

    with open(filepath) as file:
        lines = iter(file.readlines())
        for line in lines:
            if "MOTION" in line:
                break
            line = line.strip()
            if "OFFSET" in line:
                parts = line.split()
                offset = [float(parts[1]), float(parts[2]), float(parts[3])]
                offsets.append(offset)
            elif "JOINT" in line or "ROOT" in line:
                parts = line.split()
                joint = parts[1]
                joints.append(joint)
                hierarchy.append(current_parent)
                stack.append(len(joints) - 1)
                current_parent = stack[-1]
            elif "CHANNELS" in line:
                parts = line.split()
                joint_channels = parts[2:]
                channels.append(joint_channels)
            elif "End Site" in line:
                next(lines)
                next_line = next(lines).strip()
                if "OFFSET" in next_line and include_end_site:
                    parts = next_line.split()
                    offset = [float(parts[1]), float(parts[2]), float(parts[3])]
                    joints.append("End Site")
                    hierarchy.append(current_parent)
                    offsets.append(offset)
                    channels.append([])  # End sites have no channels
                next(lines)
            elif "}" in line:
                if stack:
                    stack.pop()
                    current_parent = stack[-1] if stack else -1

    motion_array, frames, frame_time = None, None, None
    if include_motion:
        motion, frames, frame_time = [], 0, 0.0
        for line in lines:
            line = line.strip()
            if not line:
                continue
            if "Frames:" in line:
                parts = line.split()
                frames = int(parts[1])
            elif "Frame Time:" in line:
                parts = line.split()
                frame_time = float(parts[2])
            else:
                frame_values = list(map(float, line.split()))
                motion.append(frame_values)
        motion_array = np.array(motion)

    if normalize:
        # Calculate the character height
        height = calculate_character_height(np.array(offsets), hierarchy)
        if height <= 0:
            print("Warning: Character height is 0, skipping normalization")
            scale_factor = 1.0
        else:
            scale_factor = 1.0 / height

        # Scale the offsets
        offsets = [np.array(offset) * scale_factor for offset in offsets]

        # Scale the motion data if it exists
        if motion_array is not None:
            # Scale only the translation channels
            for i, joint_channels in enumerate(channels):
                translation_indices = [
                    j for j, ch in enumerate(joint_channels) if ch.endswith("position")
                ]
                if translation_indices:
                    start_idx = sum(len(ch) for ch in channels[:i])
                    for idx in translation_indices:
                        motion_array[:, start_idx + idx] *= scale_factor

    return (
        np.array(offsets),
        hierarchy,
        joints,
        channels,
        motion_array,
        frames,
        frame_time,
    )


def calculate_character_height(offsets: np.ndarray, hierarchy: list[int]) -> float:
    """
    Calculate the character's height based on the joint hierarchy and offsets.

    Args:
        offsets: The local offsets of the joints.
        hierarchy: The parent index of each joint.

    Returns:
        The estimated height of the character.
    """
    max_height = 0.0
    for i, parent in enumerate(hierarchy):
        if parent != -1:
            current_height = np.linalg.norm(offsets[i])
            while parent != -1:
                current_height += np.linalg.norm(offsets[parent])
                parent = hierarchy[parent]
            max_height = max(max_height, current_height)
    return max_height


def save_bvh(
    filepath: str,
    offsets: np.ndarray,
    hierarchy: list[int],
    joints: list[str],
    motion_data: np.ndarray | None = None,
    frame_time: float | None = None,
    include_end_site: bool = False,
) -> None:
    """
    Save a BVH file with the given hierarchy, offsets, and optional motion data.

    Args:
        filepath: The path to save the BVH file.
        offsets: The local offsets of the joints.
        hierarchy: The parent index of each joint.
        joints: The names of the joints.
        motion_data: The motion data (optional).
        frame_time: Time per frame (optional, required if motion_data is provided).
        include_end_site: Whether to include End Site nodes in the output.
    """
    with open(filepath, "w") as file:
        file.write("HIERARCHY\n")
        stack = []
        motion_index = 0
        for i, joint in enumerate(joints):
            if not include_end_site and joint == "End Site":
                continue

            indent = "\t" * len(stack)
            if hierarchy[i] == -1:
                file.write(f"ROOT {joint}\n")
            elif joint == "End Site":
                file.write(f"{indent}End Site\n")
            else:
                file.write(f"{indent}JOINT {joint}\n")

            file.write(f"{indent}{{\n")
            file.write(
                f"{indent}\tOFFSET {offsets[i][0]:.6f} {offsets[i][1]:.6f} {offsets[i][2]:.6f}\n"
            )

            if joint != "End Site":
                if hierarchy[i] == -1:
                    file.write(
                        f"{indent}\tCHANNELS 6 Xposition Yposition Zposition Zrotation Yrotation Xrotation\n"
                    )
                else:
                    file.write(f"{indent}\tCHANNELS 3 Zrotation Yrotation Xrotation\n")
                motion_index += 1

            if i < len(joints) - 1 and hierarchy[i + 1] > hierarchy[i]:
                stack.append(i)
            else:
                file.write(f"{indent}}}\n")
                while stack and (i == len(joints) - 1 or hierarchy[stack[-1]] >= hierarchy[i + 1]):
                    stack.pop()
                    if stack:
                        indent = "\t" * len(stack)
                        file.write(f"{indent}}}\n")

        # Ensure the root joint (pelvis) closing bracket is written
        file.write("}\n")

        # Write motion data if provided
        if motion_data is not None and frame_time is not None:
            file.write("MOTION\n")
            file.write(f"Frames: {len(motion_data)}\n")
            file.write(f"Frame Time: {frame_time:.6f}\n")

            for i, frame in enumerate(motion_data):
                line = " ".join(f"{value:.6f}" for value in frame)
                if i < len(motion_data) - 1:
                    line += "\n"
                file.write(line)


def extract_joint_rotations(
    motion_data: np.ndarray,
    channels: list[list[str]],
    world: bool = False,
    degrees: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract joint rotations and translations from motion data.

    Args:
        motion_data: Motion data array.
        channels: List of channel types for each joint.
        world: If True, rotations are in world coordinates.
        degrees: If True, angles are in degrees.

    Returns:
        Tuple of joint rotations and translations.
    """
    rotations = []
    translations = []

    channel_index = 0
    for joint_channels in channels:
        joint_data = motion_data[:, channel_index : channel_index + len(joint_channels)]

        rot_indices = [i for i, ch in enumerate(joint_channels) if ch.endswith("rotation")]
        trans_indices = [i for i, ch in enumerate(joint_channels) if ch.endswith("position")]

        if rot_indices:
            joint_rotations = joint_data[:, rot_indices]
            rotations.append(joint_rotations)
        else:
            rotations.append(np.zeros((len(motion_data), 3)))

        if trans_indices:
            joint_translations = joint_data[:, trans_indices]
            translations.append(joint_translations)
        else:
            translations.append(np.zeros((len(motion_data), 3)))

        channel_index += len(joint_channels)

    rotations = np.array(rotations).transpose(1, 0, 2)
    translations = np.array(translations).transpose(1, 0, 2)

    return rotations, translations


def to_bvh_motion(
    rotations: Quaternion,
    root_translation: np.ndarray,
    order: str = "zyx",
    degrees: bool = True,
) -> np.ndarray:
    """
    Combine joint rotations and root translation into a single motion array.

    Args:
        rotations: Joint rotations as Quaternions.
        root_translation: Root joint translation.
        order: Rotation order for Euler angles (default: "zyx").
        degrees: If True, Euler angles are in degrees (default: True).

    Returns:
        Combined motion data array.
    """
    joint_rotations = rotations.euler(order=order, degrees=degrees).reshape(rotations.shape[0], -1)
    return np.column_stack((root_translation, joint_rotations))
