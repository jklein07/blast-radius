# Built and run on linux/amd64. The base digest is the image the tests were labeled in.
FROM python:3.13-slim@sha256:e48f59cdba69a3cbd9ed2cf0f348d90d45a8d350d98134cf1070bc106ee9ed90

ENV PYTHONDONTWRITEBYTECODE=1 PIP_DISABLE_PIP_VERSION_CHECK=1
COPY requirements.lock /app/requirements.lock
RUN pip install --no-cache-dir -r /app/requirements.lock

# sqlglot at the pinned commit, straight from its own repository
RUN python -c "import io,tarfile,urllib.request;d=urllib.request.urlopen('https://github.com/tobymao/sqlglot/archive/eb11aec4be8cb2122afe713a4816662c3f0fb2fa.tar.gz',timeout=120).read();tarfile.open(fileobj=io.BytesIO(d)).extractall('/opt',filter='data')" \
 && mv /opt/sqlglot-eb11aec4be8cb2122afe713a4816662c3f0fb2fa /opt/sqlglot

COPY *.py /app/
COPY maps /app/maps
RUN cd /opt/sqlglot && SKIP_INTEGRATION=0 PYTHONHASHSEED=0 python /app/mksnap.py /opt/sqlglot /app/snap.json \
 && python /app/prep.py /app/maps/maps.json.gz /app/maps/maps.bin \
 && python -m compileall -q -j 4 /opt/sqlglot /app /usr/local/lib/python3.13 >/dev/null 2>&1; \
    chmod -R a+rX /opt/sqlglot /app

ENV SKIP_INTEGRATION=0 PYTHONHASHSEED=0
CMD ["python", "/app/selector.py"]
