"""Monkey-patch unrealcv.Client.request() to apply a finite per-RPC timeout.

The upstream `Client.request()` in unrealcv 1.2.0 calls
`self.recv_data_q.get()` with no timeout (sitepackages/unrealcv/__init__.py:485).
If the sim silently drops an RPC response — empirically reproducible under
heavy vLLM GPU contention on `destroy(traffic_signal)` commands during
`env.reset()` — the call blocks forever, and our 600s reset budget catches it
only after the wedge is total.

This module wraps `Client.request` so each RPC waits at most
`UNREALCV_RECV_TIMEOUT_S` seconds (default 15) before raising. Failed
destroys then propagate up cleanly; `traffic_controller.reset()` either
recovers (the destroy was a no-op anyway since the signal already cleared on
the sim side) or the eval moves on to the next episode.

Import this module from evaluator.py BEFORE any code that touches
simworld_gym / unrealcv.
"""
import os
import queue


def _patch():
    import unrealcv
    Client = unrealcv.Client
    if getattr(Client, "_yz_patched", False):
        return  # idempotent

    def patched_request(self, message, timeout=5):
        """Drop-in replacement for unrealcv.Client.request() that honors a
        finite per-RPC recv timeout. Upstream signature: (message, timeout=5)
        — but upstream NEVER uses the timeout arg in the sync path; it just
        calls recv_data_q.get() with no timeout. We mirror upstream's logic
        and apply the timeout where it belongs."""
        # Async path: negative timeout → fire-and-forget
        if timeout < 0:
            if isinstance(message, list):
                self.request_batch_async(message)
            else:
                self.request_async(message)
            return True

        # Batched sync requests
        if isinstance(message, list):
            return self.request_batch(message)

        if not isinstance(message, bytes):
            message = message.encode("utf-8")

        raw_message = b"%d:%s" % (self.send_message_id, message)
        if not self.send(raw_message):
            raise ConnectionError("Failed to send: socket is closed")

        self.send_message_id += 1
        self.recv_num_q.put(-1)

        # Honor the timeout parameter. Default to 15s when caller passed the
        # upstream default of 5 — upstream 5s is unrealistic given real RPC
        # latencies (destroy of many objects + render-fence syncs). Allow
        # global override via UNREALCV_RECV_TIMEOUT_S.
        env_to = os.environ.get("UNREALCV_RECV_TIMEOUT_S")
        if env_to is not None:
            recv_timeout_s = float(env_to)
        else:
            recv_timeout_s = max(float(timeout), 15.0)
        try:
            return self.recv_data_q.get(timeout=recv_timeout_s)
        except queue.Empty:
            raise TimeoutError(
                f"unrealcv RPC timed out after {recv_timeout_s}s "
                f"(no response to: {raw_message[:80]!r})"
            )

    Client.request = patched_request
    Client._yz_patched = True


_patch()
