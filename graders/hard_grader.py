def score(logs: list[dict]) -> float:
    """
    Grader for the Hard Task: Full Optimization.
    Goal is to maximize flow while ensuring rule adherence (0 violations).
    """
    total_flow = 0
    violations = 0
    
    for step in logs:
        if "info" in step and "flow" in step["info"]:
            total_flow += step["info"]["flow"]
        if "observation" in step and "violations" in step["observation"]:
            # Capture strictly the highest violations count assuming cumulative
            violations = max(violations, step["observation"]["violations"])
            
    # Max Flow normalized roughly against expected throughput over steps (50+)
    flow_score = min(1.0, total_flow / 50.0) 
    
    # Violation score (10.0 is MAX_VIOLATIONS)
    violation_score = max(0.0, 1.0 - (violations / 10.0))
    
    composite_score = (flow_score * 0.5) + (violation_score * 0.5)
    return max(0.0, min(1.0, composite_score))
