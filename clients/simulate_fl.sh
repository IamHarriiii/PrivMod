#!/bin/bash
for i in {1..3}; do
  docker run --name client$i -d fl-client --cid $i
done