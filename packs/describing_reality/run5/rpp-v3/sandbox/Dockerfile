# Run 5 sandbox for model-written code (V3-043). One throwaway container per tool call; see v3/sandbox.py for the
# run flags (no network, read-only, tmpfs scratch, memory/CPU/process limits, non-root, no privileges).
# Only the standard library and numpy are available to model code, as in v2 (a3_sim.py, g1c1.py).
FROM python:3.12.10-slim

RUN pip install --no-cache-dir numpy==2.5.4 \
 && useradd --uid 1000 --no-create-home --shell /usr/sbin/nologin sandbox \
 && rm -rf /root/.cache

# sandbox_runner.py is a copy of rpp-core/harness/sandbox_runner.py (the v2 audit-hook runner), the second layer.
COPY sandbox_runner.py entry.py /opt/sandbox/
RUN chmod 0444 /opt/sandbox/*.py

USER 1000
WORKDIR /scratch
ENV PYTHONIOENCODING=utf-8 PYTHONDONTWRITEBYTECODE=1
ENTRYPOINT ["python", "-I", "/opt/sandbox/entry.py"]
