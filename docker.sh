#!/bin/bash
docker run -it --rm --runtime=nvidia -v $(pwd):/workspace eslexoro/scanx:pytorch
