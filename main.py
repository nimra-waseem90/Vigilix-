from fusion.fusion_engine import FusionEngine
from detection.fire_detection import detect_fire
from detection.weapon_detection import detect_weapon


FIRE_VIDEO = "data/raw/videos/fire_test.mp4"
WEAPON_VIDEO = "data/raw/videos/weapon_test.mp4"


def process_event(fusion, event):

    if event is None:
        print("\nNo event detected.")
        return

    print("\n----------------------------------------")
    print("EVENT RECEIVED")
    print("----------------------------------------")
    print(event)

    fused_alert = fusion.add_event(event)

    if fused_alert:

        print("\n========================================")
        print("        VIGILIX FUSION ALERT")
        print("========================================")

        print(
            f"Event      : {fused_alert['event']}"
        )

        print(
            f"Label      : {fused_alert['label']}"
        )

        print(
            f"Confidence : "
            f"{fused_alert['confidence']}"
        )

        print(
            f"Severity   : "
            f"{fused_alert['severity']}"
        )

        print(
            f"Timestamp  : "
            f"{fused_alert['timestamp']}"
        )

        print(
            "\nSource Events:"
        )

        for source_event in fused_alert[
            "source_events"
        ]:
            print(source_event)

        print("========================================")


def main():

    print("\n========================================")
    print("          VIGILIX FUSION SYSTEM")
    print("========================================")

    fusion = FusionEngine(
        time_window=5.0
    )

    # ------------------------------------
    # FIRE DETECTION
    # ------------------------------------

    print("\n🔥 Starting Fire Detection...")

    fire_event = detect_fire(
        FIRE_VIDEO
    )

    process_event(
    fusion,
    fire_event
)

# Fire and weapon are currently
# being tested using separate videos.
# Clear the previous event before
# starting the next independent test.

    fusion.clear_events()

# ------------------------------------
# WEAPON DETECTION
# ------------------------------------

    print("\n🔫 Starting Weapon Detection...")

    weapon_event = detect_weapon(
        WEAPON_VIDEO
    )

    process_event(
        fusion,
        weapon_event
    )

    print("\n========================================")
    print("       VIGILIX FUSION TEST COMPLETE")
    print("========================================")


if __name__ == "__main__":
    main()