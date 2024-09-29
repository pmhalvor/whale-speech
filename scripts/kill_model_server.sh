echo "Checking for running model_server.py files..."
ps aux | grep python | grep whale | grep model_server.py

PID=$(ps aux | grep python | grep whale | grep model_server.py | awk '{print $2}')
if [ -z "$PID" ]; then
    echo "No model_server.py files running."
else
    echo "Killing PID: $PID"
    kill -9 $PID
    sleep 2

    echo "Checking for running model_server.py files..."
    ps aux | grep python | grep whale | grep model_server.py
fi

echo "Done."