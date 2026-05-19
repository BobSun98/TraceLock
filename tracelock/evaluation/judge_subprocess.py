from __future__ import annotations

import concurrent.futures
import json
import os
import queue
import subprocess
import threading
from collections import deque
from pathlib import Path
from typing import Any

from tqdm.auto import tqdm


WORKER_PATH = Path(__file__).resolve().with_name("judge_worker.py")


class PersistentJudgeProcess:
    def __init__(
        self,
        *,
        judge_python: str,
        device: str | None,
        model_name: str,
        torch_dtype_name: str,
        temperature: float,
        max_input_length: int,
        max_new_tokens: int,
    ) -> None:
        command = [
            judge_python,
            str(WORKER_PATH),
            "--serve",
            "--model-name",
            model_name,
            "--torch-dtype",
            torch_dtype_name,
            "--temperature",
            str(temperature),
            "--max-input-length",
            str(max_input_length),
            "--max-new-tokens",
            str(max_new_tokens),
        ]
        env = os.environ.copy()
        if device:
            env["CUDA_VISIBLE_DEVICES"] = device.split(":", 1)[1] if device.startswith("cuda:") else device

        self._stderr_tail: deque[str] = deque(maxlen=200)
        self._stderr_lock = threading.Lock()
        self.process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            env=env,
        )
        self.command = command
        self.device = device
        self._stderr_thread = threading.Thread(target=self._drain_stderr, daemon=True)
        self._stderr_thread.start()

        print(
            json.dumps(
                {
                    "stage": "judge_worker_spawned",
                    "device": device,
                    "pid": self.process.pid,
                    "command": command,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    def _drain_stderr(self) -> None:
        assert self.process.stderr is not None
        for line in self.process.stderr:
            with self._stderr_lock:
                self._stderr_tail.append(line.rstrip("\n"))

    def stderr_tail(self) -> str:
        with self._stderr_lock:
            return "\n".join(self._stderr_tail)

    def run_batch(self, items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if self.process.poll() is not None:
            raise RuntimeError(
                f"Judge worker exited early with code {self.process.returncode}.\n"
                f"Command: {' '.join(self.command)}\n"
                f"Stderr:\n{self.stderr_tail()}"
            )
        assert self.process.stdin is not None
        assert self.process.stdout is not None

        self.process.stdin.write(json.dumps({"items": items}, ensure_ascii=False) + "\n")
        self.process.stdin.flush()
        response_line = self.process.stdout.readline()
        if not response_line:
            raise RuntimeError(
                "Judge worker closed stdout unexpectedly.\n"
                f"Command: {' '.join(self.command)}\n"
                f"Stderr:\n{self.stderr_tail()}"
            )
        payload = json.loads(response_line)
        return [dict(item) for item in payload["results"]]

    def close(self) -> None:
        if self.process.poll() is None:
            try:
                assert self.process.stdin is not None
                self.process.stdin.write(json.dumps({"cmd": "shutdown"}) + "\n")
                self.process.stdin.flush()
                self.process.stdin.close()
            except Exception:
                self.process.terminate()
        returncode = self.process.wait(timeout=30)
        if returncode != 0:
            raise RuntimeError(
                f"Judge worker exited with code {returncode}.\n"
                f"Command: {' '.join(self.command)}\n"
                f"Stderr:\n{self.stderr_tail()}"
            )


def run_with_subprocess(
    *,
    judge_python: str,
    items: list[dict[str, Any]],
    batch_size: int,
    model_name: str,
    torch_dtype_name: str,
    temperature: float,
    max_input_length: int,
    max_new_tokens: int,
    cuda_visible_devices: str | None = None,
    gpu_devices: list[str] | None = None,
    show_progress: bool = False,
) -> list[dict[str, Any]]:
    if not items:
        return []
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")

    normalized_devices = [str(device) for device in (gpu_devices or [])]
    if not normalized_devices:
        normalized_devices = [cuda_visible_devices] if cuda_visible_devices is not None else [None]

    worker_count = min(len(normalized_devices), max(1, len(items)))
    worker_devices = normalized_devices[:worker_count]
    batches = [items[start : start + batch_size] for start in range(0, len(items), batch_size)]
    task_queue: queue.Queue[list[dict[str, Any]]] = queue.Queue()
    for batch in batches:
        task_queue.put(batch)

    print(
        json.dumps(
            {
                "stage": "judge_dispatch",
                "judge_python": judge_python,
                "model_name": model_name,
                "worker_count": worker_count,
                "gpu_devices": worker_devices,
                "items": len(items),
                "batch_size": batch_size,
                "num_batches": len(batches),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    results: list[dict[str, Any]] = []
    progress = tqdm(total=len(items), desc="judge", unit="item", dynamic_ncols=True) if show_progress else None
    progress_lock = threading.Lock()

    def consume_batches(device: str | None) -> list[dict[str, Any]]:
        worker = PersistentJudgeProcess(
            judge_python=judge_python,
            device=device,
            model_name=model_name,
            torch_dtype_name=torch_dtype_name,
            temperature=temperature,
            max_input_length=max_input_length,
            max_new_tokens=max_new_tokens,
        )
        device_results: list[dict[str, Any]] = []
        try:
            while True:
                try:
                    batch = task_queue.get_nowait()
                except queue.Empty:
                    break
                batch_results = worker.run_batch(batch)
                device_results.extend(batch_results)
                if progress is not None:
                    with progress_lock:
                        progress.update(len(batch_results))
            return device_results
        finally:
            worker.close()

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=worker_count) as executor:
            future_map = {
                executor.submit(consume_batches, device): device
                for device in worker_devices
            }
            for future in concurrent.futures.as_completed(future_map):
                device = future_map[future]
                device_results = future.result()
                results.extend(device_results)
                print(
                    json.dumps(
                        {
                            "stage": "judge_worker_done",
                            "device": device,
                            "returned_items": len(device_results),
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
    finally:
        if progress is not None:
            progress.close()
    return results
