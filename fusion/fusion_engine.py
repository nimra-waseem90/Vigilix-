from datetime import datetime


class FusionEngine:

    def __init__(self, time_window=5.0):

        self.time_window = time_window
        self.recent_events = []

    # ========================================================
    # ADD EVENT
    # ========================================================

    def add_event(self, event):

        if not event:
            return None

        if not event.get("detected", False):
            return None

        self.recent_events.append(event)

        self._remove_old_events(
            event["timestamp"]
        )

        return self._fuse_events()

    # ========================================================
    # REMOVE OLD EVENTS
    # ========================================================

    def _remove_old_events(self, current_timestamp):

        self.recent_events = [
            event
            for event in self.recent_events
            if current_timestamp - event["timestamp"]
            <= self.time_window
        ]

    # ========================================================
    # FUSION LOGIC
    # ========================================================

    def _fuse_events(self):

        if not self.recent_events:
            return None

        event_types = {
            event["event"]
            for event in self.recent_events
        }

        # ----------------------------------------------------
        # WEAPON + GUNSHOT
        # ----------------------------------------------------

        if (
            "weapon" in event_types
            and "gunshot" in event_types
        ):

            return self._create_fused_alert(
                "weapon_gunshot",
                "Weapon + Gunshot",
                "CRITICAL"
            )

        # ----------------------------------------------------
        # FIRE + SMOKE ALARM
        # ----------------------------------------------------

        if (
            "fire" in event_types
            and "smoke_alarm" in event_types
        ):

            return self._create_fused_alert(
                "fire_smoke_alarm",
                "Fire + Smoke Alarm",
                "CRITICAL"
            )

        # ----------------------------------------------------
        # FIRE + ALARM
        # ----------------------------------------------------

        if (
            "fire" in event_types
            and "alarm" in event_types
        ):

            return self._create_fused_alert(
                "fire_alarm",
                "Fire + Alarm",
                "CRITICAL"
            )

        # ----------------------------------------------------
        # MULTIPLE DIFFERENT EVENTS
        # ----------------------------------------------------

        if len(event_types) > 1:

            return self._create_fused_alert(
                "multiple_events",
                "Multiple Suspicious Events",
                "HIGH"
            )

        # ----------------------------------------------------
        # SINGLE FIRE
        # ----------------------------------------------------

        if event_types == {"fire"}:

            return self._create_fused_alert(
                "fire",
                "Fire Detected",
                "HIGH"
            )

        # ----------------------------------------------------
        # SINGLE WEAPON
        # ----------------------------------------------------

        if event_types == {"weapon"}:

            return self._create_fused_alert(
                "weapon",
                "Weapon Detected",
                "HIGH"
            )

        # ----------------------------------------------------
        # SINGLE AUDIO EVENT
        # ----------------------------------------------------

        return self._create_fused_alert(
            "audio_event",
            "Suspicious Audio Detected",
            "MEDIUM"
        )

    # ========================================================
    # CREATE FUSED ALERT
    # ========================================================

    def _create_fused_alert(
        self,
        event,
        label,
        severity
    ):

        confidence = max(
            event_data["confidence"]
            for event_data in self.recent_events
        )

        timestamp = max(
            event_data["timestamp"]
            for event_data in self.recent_events
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
            "source_events": (
                self.recent_events.copy()
            ),
            "created_at": (
                datetime.now().isoformat()
            )
        }

    # ========================================================
    # CLEAR EVENTS
    # ========================================================

    def clear_events(self):

        self.recent_events.clear()


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    fusion = FusionEngine(
        time_window=5.0
    )

    weapon_event = {
        "detected": True,
        "event": "weapon",
        "label": "Gun",
        "confidence": 0.91,
        "timestamp": 15.4
    }

    gunshot_event = {
        "detected": True,
        "event": "gunshot",
        "label": "Gunshot",
        "confidence": 0.78,
        "timestamp": 15.7
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