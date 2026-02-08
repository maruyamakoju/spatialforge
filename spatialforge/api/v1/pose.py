"""/v1/pose — Camera pose estimation from multiple images or video."""

from __future__ import annotations

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile

from ...auth.api_keys import APIKeyRecord, get_current_user
from ...models.responses import CameraIntrinsics, CameraPose, PoseResponse

router = APIRouter()

MAX_FILE_SIZE = 100 * 1024 * 1024  # 100 MB for video


@router.post("/pose", response_model=PoseResponse)
async def estimate_pose(
    request: Request,
    video: UploadFile = File(None, description="Video file (MP4)"),
    images: list[UploadFile] = File(None, description="Multiple image files"),
    output_pointcloud: bool = Form(False),
    user: APIKeyRecord = Depends(get_current_user),
):
    """Estimate camera poses from a video or multiple images.

    Returns camera intrinsics (fx, fy, cx, cy) and extrinsics (rotation, translation)
    for each frame/image, plus an optional sparse point cloud.
    """
    frames = []

    if video is not None:
        content = await video.read()
        if len(content) > MAX_FILE_SIZE:
            raise HTTPException(status_code=413, detail="Video exceeds 100MB limit")

        from ...utils.video import extract_keyframes, save_uploaded_video, validate_video

        path = save_uploaded_video(content)
        try:
            validate_video(path)
            frames = extract_keyframes(path, target_fps=2.0)
        finally:
            import os

            os.unlink(path)
    elif images is not None and len(images) >= 2:
        from ...utils.image import load_image_rgb

        for img_file in images:
            data = await img_file.read()
            try:
                rgb = load_image_rgb(data)
                frames.append(rgb)
            except ValueError as e:
                raise HTTPException(status_code=400, detail=f"Invalid image: {e}")
    else:
        raise HTTPException(
            status_code=400,
            detail="Provide either a video file or at least 2 images",
        )

    if len(frames) < 2:
        raise HTTPException(status_code=400, detail="Need at least 2 frames for pose estimation")

    # Run pose estimation
    from ...inference.pose_engine import PoseEngine

    engine = PoseEngine(request.app.state.model_manager)
    result = engine.estimate_poses(frames, output_pointcloud=output_pointcloud)

    # Upload point cloud if requested
    pointcloud_url = None
    if result.pointcloud is not None and request.app.state.object_store is not None:
        import io

        import numpy as np

        buf = io.BytesIO()
        np.save(buf, result.pointcloud)
        key = request.app.state.object_store.upload_bytes(
            buf.getvalue(),
            content_type="application/octet-stream",
            prefix="pointclouds",
            extension="npy",
        )
        pointcloud_url = request.app.state.object_store.get_presigned_url(key)

    # Build response
    camera_poses = []
    for p in result.poses:
        camera_poses.append(
            CameraPose(
                frame_index=p.frame_index,
                rotation=p.rotation.tolist(),
                translation=p.translation.tolist(),
                intrinsics=CameraIntrinsics(
                    fx=p.fx,
                    fy=p.fy,
                    cx=p.cx,
                    cy=p.cy,
                    width=p.width,
                    height=p.height,
                ),
            )
        )

    return PoseResponse(
        camera_poses=camera_poses,
        pointcloud_url=pointcloud_url,
        num_frames=len(camera_poses),
        processing_time_ms=round(result.processing_time_ms, 2),
    )
