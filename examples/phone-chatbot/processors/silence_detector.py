"""Silence detector for Pipecat pipelines.

This processor watches incoming frames for user activity. If the user is
quiet for `timeout` seconds we send a gentle TTS prompt (through the LLM).
After `max_retries` unanswered prompts we end the task to hang‑up the call.
All comments are kept very short and in plain English as requested.
"""

import asyncio
from time import monotonic

from pipecat.frames.frames import TranscriptionFrame, LLMMessagesFrame, EndTaskFrame
from pipecat.processors.frame_processor import FrameProcessor, FrameDirection


class SilenceDetector(FrameProcessor):
    """Detect long silence, prompt user, optionally terminate call."""

    def __init__(self, tts_prompt: str, llm, timeout: int = 10, max_retries: int = 3):
        super().__init__()
        self.tts_prompt = tts_prompt          # What we say after silence
        self.llm = llm                        # LLM context aggregator ref
        self.timeout = timeout                # Seconds before we act
        self.max_retries = max_retries        # Times we prompt before hang‑up
        self.last_heard = monotonic()         # Last time we got speech
        self.missed = 0                       # How many prompts already sent

    async def process_frame(self, frame, direction):
        """Intercept frames and reset timer on user speech."""
        await super().process_frame(frame, direction)

        # User spoke => reset counters
        if isinstance(frame, TranscriptionFrame) and frame.text.strip():
            if hasattr(self.llm, "session_manager"):
                self.llm.session_manager.silence_events += 1
            self.last_heard = monotonic()
            self.missed = 0

        # Always forward the frame downstream/upstream
        await self.push_frame(frame, direction)

    async def on_no_activity(self):
        """Background loop that checks for silence every second."""
        while True:
            await asyncio.sleep(1)
            if monotonic() - self.last_heard > self.timeout:
                self.missed += 1
                # Send TTS prompt through LLM
                await self.llm.queue_frame(
                    LLMMessagesFrame([
                        {"role": "system", "content": self.tts_prompt}
                    ]),
                    FrameDirection.UPSTREAM,
                )
                self.last_heard = monotonic()  # Reset timer after prompt

                # Hang up if user stays silent too many times
                if self.missed >= self.max_retries:
                    await self.llm.queue_frame(
                        EndTaskFrame(), FrameDirection.UPSTREAM
                    )
