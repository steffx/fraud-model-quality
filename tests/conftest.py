"""Shared fixtures. Training runs once per test session and is reused by every test."""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import pytest

from fraudmodel.data import generate_transactions
from fraudmodel.train import save_artifacts, train_all

SEED = 42


@pytest.fixture(scope="session")
def dataset():
    return generate_transactions(seed=SEED)


@pytest.fixture(scope="session")
def run(dataset):
    """All candidate models trained on the standard dataset."""
    return train_all(dataset, seed=SEED)


@pytest.fixture(scope="session")
def champion(run):
    return run["results"][run["champion"]]


@pytest.fixture(scope="session")
def artifacts_dir(run, tmp_path_factory) -> Path:
    out = tmp_path_factory.mktemp("artifacts")
    save_artifacts(run, out)
    return out


@pytest.fixture(scope="session")
def client(artifacts_dir):
    from fastapi.testclient import TestClient

    from fraudmodel.service import create_app

    return TestClient(create_app(artifacts_dir))


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="session")
def live_server(artifacts_dir):
    """Runs the real service with uvicorn for the Playwright tests."""
    port = _free_port()
    env = {**os.environ, "ARTIFACTS_DIR": str(artifacts_dir)}
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "fraudmodel.service:create_app", "--factory", "--port", str(port)],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    url = f"http://127.0.0.1:{port}"
    for _ in range(100):
        try:
            urllib.request.urlopen(f"{url}/health", timeout=1)
            break
        except OSError:
            if proc.poll() is not None:
                raise RuntimeError(proc.stderr.read().decode())
            time.sleep(0.1)
    yield url
    proc.terminate()
    proc.wait(timeout=10)


@pytest.fixture(scope="session")
def browser_type_launch_args(browser_type_launch_args):
    """Allows a custom Chromium binary (CHROMIUM_PATH), e.g. behind a strict corporate proxy."""
    path = os.environ.get("CHROMIUM_PATH")
    return {**browser_type_launch_args, **({"executable_path": path} if path else {})}
