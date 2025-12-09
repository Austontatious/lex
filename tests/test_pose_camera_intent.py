from backend.lexi.utils.pose_camera_intent import classify_pose_camera


def test_classify_pose_and_camera_from_text():
    text = "she is lounging on the couch, camera slightly above, soft high angle"
    intent = classify_pose_camera(text)
    assert intent.pose_bucket == "POSE_7"
    assert intent.camera_bucket in {"CAM_SLIGHT_HIGH", "CAM_HIGH"}
    assert intent.pose_feel in {"confident", "playful", "cozy"}
