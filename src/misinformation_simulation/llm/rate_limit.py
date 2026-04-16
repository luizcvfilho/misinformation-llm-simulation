from __future__ import annotations

import time


class MinuteRateLimiter:
    def __init__(self, max_requests_per_minute: int | None) -> None:
        self.max_requests_per_minute = max_requests_per_minute
        self.current_minute_bucket = int(time.time() // 60)
        self.requests_in_current_minute = 0

    def acquire(self) -> None:
        if self.max_requests_per_minute is None:
            return

        now_bucket = int(time.time() // 60)
        if now_bucket != self.current_minute_bucket:
            self.current_minute_bucket = now_bucket
            self.requests_in_current_minute = 0

        if self.requests_in_current_minute >= self.max_requests_per_minute:
            seconds_until_next_minute = 60 - (time.time() % 60)
            time.sleep(seconds_until_next_minute)
            self.current_minute_bucket = int(time.time() // 60)
            self.requests_in_current_minute = 0

        self.requests_in_current_minute += 1
