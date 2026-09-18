from datetime import datetime


class FusionEngine:

    def __init__(self, time_window=5.0):
        """
        time_window:
            Maximum number of seconds allowed between
            related detections.

        Events are also separated by camera_id so that
        detections from different cameras are not fused.
        """

        self.time_window = time_window
        self.recent_events = []

    # ==========================================================
    # ADD EVENT
    # ==========================================================

    def add_event(self, event):

        if not event:
            return None

        # Ignore "no detection" events
        if not event.get("detected", False):
            return None

        # Make sure required fields exist
        if "timestamp" not in event:
            return None

        self.recent_events.append(event)

        self._remove_old_events(
            event["timestamp"]
        )

        return self._fuse_events(
            camera_id=event.get(
                "camera_id",
                "default"
            )
        )

    # ==========================================================
    # REMOVE OLD EVENTS
    # ==========================================================

    def _remove_old_events(self, current_timestamp):

        self.recent_events = [
            event
            for event in self.recent_events
            if abs(
                current_timestamp - event["timestamp"]
            ) <= self.time_window
        ]

    # ==========================================================
    # GET EVENTS FOR SAME CAMERA
    # ==========================================================

    def _get_camera_events(self, camera_id):

        return [
            event
            for event in self.recent_events
            if event.get(
                "camera_id",
                "default"
            ) == camera_id
        ]

    # ==========================================================
    # NORMALIZE EVENT TYPE
    # ==========================================================

    def _get_event_type(self, event):

        event_type = event.get(
            "event",
            ""
        ).lower()

        label = str(
            event.get(
                "label",
                ""
            )
        ).lower()

        # ------------------------------
        # WEAPON
        # ------------------------------

        if event_type == "weapon":
            return "weapon"

        # ------------------------------
        # FIRE
        # ------------------------------

        if event_type == "fire":
            return "fire"

        # ------------------------------
        # GUNSHOT / AUDIO
        # ------------------------------

        if event_type in [
            "gunshot",
            "audio"
        ]:

            if (
                "gunshot" in label
                or "gunfire" in label
                or "machine gun" in label
            ):
                return "gunshot"

        # ------------------------------
        # SMOKE ALARM
        # ------------------------------

        if (
            event_type == "smoke_alarm"
            or "smoke detector" in label
        ):
            return "smoke_alarm"

        # ------------------------------
        # ALARM
        # ------------------------------

        if event_type == "alarm":
            return "alarm"

        if "alarm" in label:
            return "alarm"

        # ------------------------------
        # EXPLOSION
        # ------------------------------

        if (
            event_type == "explosion"
            or "explosion" in label
        ):
            return "explosion"

        # ------------------------------
        # SCREAM / SHOUT
        # ------------------------------

        if (
            event_type in [
                "scream",
                "shout"
            ]
            or "scream" in label
            or "shout" in label
        ):
            return "scream"

        # ------------------------------
        # GLASS
        # ------------------------------

        if (
            event_type == "glass"
            or "glass" in label
        ):
            return "glass"

        # ------------------------------
        # UNKNOWN AUDIO
        # ------------------------------

        if event_type == "audio":
            return "audio"

        return event_type

    # ==========================================================
    # FUSION LOGIC
    # ==========================================================

    def _fuse_events(self, camera_id):

        camera_events = self._get_camera_events(
            camera_id
        )

        if not camera_events:
            return None

        event_types = {
            self._get_event_type(event)
            for event in camera_events
        }

        # ======================================================
        # WEAPON + GUNSHOT
        # ======================================================

        if (
            "weapon" in event_types
            and "gunshot" in event_types
        ):

            return self._create_fused_alert(
                camera_events,
                "weapon_gunshot",
                "Weapon + Gunshot",
                "CRITICAL"
            )

        # ======================================================
        # FIRE + SMOKE ALARM
        # ======================================================

        if (
            "fire" in event_types
            and "smoke_alarm" in event_types
        ):

            return self._create_fused_alert(
                camera_events,
                "fire_smoke_alarm",
                "Fire + Smoke Alarm",
                "CRITICAL"
            )

        # ======================================================
        # FIRE + ALARM
        # ======================================================

        if (
            "fire" in event_types
            and "alarm" in event_types
        ):

            return self._create_fused_alert(
                camera_events,
                "fire_alarm",
                "Fire + Alarm",
                "CRITICAL"
            )

        # ======================================================
        # FIRE + EXPLOSION
        # ======================================================

        if (
            "fire" in event_types
            and "explosion" in event_types
        ):

            return self._create_fused_alert(
                camera_events,
                "fire_explosion",
                "Fire + Explosion",
                "CRITICAL"
            )

        # ======================================================
        # MULTIPLE SUSPICIOUS EVENTS
        # ======================================================

        if len(event_types) > 1:

            return self._create_fused_alert(
                camera_events,
                "multiple_events",
                "Multiple Suspicious Events",
                "HIGH"
            )

        # ======================================================
        # FIRE ONLY
        # ======================================================

        if event_types == {"fire"}:

            return self._create_fused_alert(
                camera_events,
                "fire",
                "Fire Detected",
                "HIGH"
            )

        # ======================================================
        # WEAPON ONLY
        # ======================================================

        if event_types == {"weapon"}:

            return self._create_fused_alert(
                camera_events,
                "weapon",
                "Weapon Detected",
                "HIGH"
            )

        # ======================================================
        # GUNSHOT ONLY
        # ======================================================

        if event_types == {"gunshot"}:

            return self._create_fused_alert(
                camera_events,
                "gunshot",
                "Gunshot Detected",
                "HIGH"
            )

        # ======================================================
        # EXPLOSION ONLY
        # ======================================================

        if event_types == {"explosion"}:

            return self._create_fused_alert(
                camera_events,
                "explosion",
                "Explosion Detected",
                "HIGH"
            )

        # ======================================================
        # OTHER AUDIO
        # ======================================================

        if "audio" in event_types:

            return self._create_fused_alert(
                camera_events,
                "audio_event",
                "Suspicious Audio Detected",
                "MEDIUM"
            )

        return None

    # ==========================================================
    # CREATE FUSED ALERT
    # ==========================================================

    def _create_fused_alert(
        self,
        source_events,
        event,
        label,
        severity
    ):

        confidence = max(
            event_data["confidence"]
            for event_data in source_events
        )

        timestamp = max(
            event_data["timestamp"]
            for event_data in source_events
        )

        camera_id = source_events[0].get(
            "camera_id",
            "default"
        )

        return {
            "fused": True,
            "event": event,
            "label": label,
            "confidence": round(
                confidence,
                3
            ),
            "severity": severity,
            "timestamp": timestamp,
            "camera_id": camera_id,
            "source_events": source_events.copy(),
            "created_at": datetime.now().isoformat()
        }

    # ==========================================================
    # CLEAR EVENTS
    # ==========================================================

    def clear_events(self):

        self.recent_events.clear()


# ==============================================================
# TEST
# ==============================================================

if __name__ == "__main__":

    fusion = FusionEngine(
        time_window=5.0
    )

    weapon_event = {
        "detected": True,
        "event": "weapon",
        "label": "Gun",
        "confidence": 0.91,
        "timestamp": 15.4,
        "source": "rtsp",
        "camera_id": "camera_01"
    }

    gunshot_event = {
        "detected": True,
        "event": "audio",
        "label": "Gunshot, gunfire",
        "confidence": 0.145,
        "timestamp": 15.7,
        "source": "rtsp_audio",
        "camera_id": "camera_01"
    }

    print("\nAdding weapon event...")

    result = fusion.add_event(
        weapon_event
    )

    print(result)

    print("\nAdding gunshot event...")

    result = fusion.add_event(
        gunshot_event
    )

    print("\nFUSION RESULT:")

    print(result)

