#!/bin/bash
# PORT=${1};
# echo "Establishing server on port $PORT"
nohup python -m http.server --directory tech-blogs 6688 >/dev/null 2>&1 & # runs in background, still doesn't create nohup.out
echo "Serving HTTP on 0.0.0.0 port 6688 (http://0.0.0.0:6688/) ..."
echo "Http server established in the background"