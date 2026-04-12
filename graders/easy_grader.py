def score(logs: list[dict]) -> float:
    """
    Grader for the Easy Task: Green Wave Timing.
    Goal is to clear the initial heavily congested queue (15 vehicles).
    """
    total_flow = 0
    for step in logs:
        if "info" in step and "flow" in step["info"]:
            total_flow += step["info"]["flow"]
            
    # Normalize score by clearing the total initial vehicles
    flow_score = min(0.99, max(0.01, total_flow / 15.0))
    return flow_score
