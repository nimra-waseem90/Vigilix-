from datetime import datetime
def create_event(
    event,
    label,
    confidence,
    timestamp,
    source="unknown",
    camera_id="default",
    detected=True
):
    """
    Creates a standardized Vigilix detection event.
    """

    return {
        "detected": detected,
        "event": event,
        "label": label,
        "confidence": round(float(confidence), 3),
        "timestamp": float(timestamp),
        "source": source,
        "camera_id": camera_id,
        "created_at": datetime.now().isoformat()
    }


def no_event(
    event,
    timestamp,
    source="unknown",
    camera_id="default"
):
    """
    Creates a standardized 'no detection' event.
    """

    return {
        "detected": False,
        "event": event,
        "label": None,
        "confidence": 0.0,
        "timestamp": float(timestamp),
        "source": source,
        "camera_id": camera_id,
        "created_at": datetime.now().isoformat()
    }