def score(logs: list[dict]) -> float:
    """
    Grader for the Medium Task: Congestion Reduction.
    Goal is to minimize the average wait time across all queues.
    """
    wait_times = []
    for step in logs:
        if "observation" in step and "avg_wait" in step["observation"]:
            wait_times.append(step["observation"]["avg_wait"])
            
    if not wait_times:
        return 0.0
        
    avg_wait = sum(wait_times) / len(wait_times)
    
    # Lower wait time translates to higher score (50.0 is MAX_WAIT)
    wait_score = max(0.0, 1.0 - (avg_wait / 50.0))
    return wait_score
