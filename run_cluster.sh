#!/usr/bin/env bash
# docker build -t ls6-staff-registry.informatik.uni-wuerzburg.de/koopmann/comptrails-dataset:latest -f Code/data_generation/Dockerfile .
# docker push ls6-staff-registry.informatik.uni-wuerzburg.de/koopmann/comptrails-dataset:latest
# fastbuildah bud -t ls6-staff-registry.informatik.uni-wuerzburg.de/koopmann/hydras:latest -f kubernetes/Dockerfile .
# fastbuildah push ls6-staff-registry.informatik.uni-wuerzburg.de/koopmann/hydras:latest
kubectl -n koopmann delete job hydras-$1
kubectl -n koopmann apply -f kubernetes/$1.yml
