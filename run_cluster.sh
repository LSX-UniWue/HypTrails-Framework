#!/usr/bin/env bash
docker buildx build --platform linux/amd64 --load -t ls6-staff-registry.informatik.uni-wuerzburg.de/koopmann/hydras:latest -f kubernetes/Dockerfile .
docker push ls6-staff-registry.informatik.uni-wuerzburg.de/koopmann/hydras:latest
# fastbuildah bud -t ls6-staff-registry.informatik.uni-wuerzburg.de/koopmann/hydras:latest -f kubernetes/Dockerfile .
# fastbuildah push ls6-staff-registry.informatik.uni-wuerzburg.de/koopmann/hydras:latest
kubectl -n koopmann delete job hydras-$1
kubectl -n koopmann apply -f kubernetes/$1.yml
