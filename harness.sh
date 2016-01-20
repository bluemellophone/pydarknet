mkdir results
mkdir results/nms-test
mkdir results/all-test

for f in test/*; do
    ./darknet yolo test cfg/yolo.cfg yolo.weights "$f"
    mv predictions.png "results/nms-$f"
    ./darknet yolo test cfg/yolo.cfg yolo.weights "$f" -thresh 0
    mv predictions.png "results/all-$f"
done
