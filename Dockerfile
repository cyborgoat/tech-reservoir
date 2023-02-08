FROM python:3.11-alpine3.16
LABEL maintainer="cyborgoat.com"

ENV PYTHONUNBUFFERED 1

COPY ./requirements.txt /tmp/requirements.txt
COPY ./requirements.dev.txt /tmp/requirements.dev.txt
COPY ./app /app
COPY ./tech-blog /home/data/tech-blog
COPY ./scripts /scripts
WORKDIR /app

ARG DEV=false

RUN sed -i 's/dl-cdn.alpinelinux.org/mirrors.tuna.tsinghua.edu.cn/g' /etc/apk/repositories
RUN apk add --no-cache gcc musl-dev linux-headers
RUN pip3 config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip3 config set install.trusted-host pypi.tuna.tsinghua.edu.cn

RUN python -m venv /py && \
    /py/bin/pip install --upgrade pip && \
    /py/bin/pip install -r /tmp/requirements.txt && \
    if [ $DEV = "true" ]; \
    then /py/bin/pip install -r /tmp/requirements.dev.txt ; \
    fi && \
    rm -rf /tmp && \
    adduser \
    --disabled-password \
    --no-create-home \
    cyborgoat && \
    mkdir -p /vol/web/media && \
    mkdir -p /vol/web/static && \
    chown -R cyborgoat:cyborgoat /vol && \
    chmod -R 755 /vol && \
    chmod -R +x /scripts

USER cyborgoat

ENV PATH="/scripts:/py/bin:$PATH"

CMD ["run.sh"]