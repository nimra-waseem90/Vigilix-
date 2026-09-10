from fusion.event_format import create_event


event = create_event(
    event="weapon",
    label="Gun",
    confidence=0.91,
    timestamp=15.4
)

print("\nVIGILIX EVENT")
print(event)