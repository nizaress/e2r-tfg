PID=$(lsof -ti :8000)
if [ -n "$PID" ]; then
    echo "Matando proceso en el puerto 8000 (PID: $PID)"
    kill -9 $PID
fi

xdg-open interfaz.html
uvicorn main:app --reload --host 0.0.0.0 --port 8000
