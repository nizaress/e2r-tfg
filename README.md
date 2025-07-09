# TFG sobre Lectura Fácil (E2R)

Este es el repositorio con el código implementado como apoyo para la realización del Trabajo de Fin de Grado llamado "Sistema de IA generativa para lectura facilitada". Podemos encontrarlo aquí como "tfg.pdf".

## Instalaciones previas
Estas son las librerías que debemos instalar en caso de querer utilizar la interfaz de inferencia.

```bash
pip install fastapi uvicorn pydantic huggingface_hub[inference] transformers peft torch --quiet
```

## Uso
Para usar la interfaz solamente debemos ejecutar lo siguiente en la ubicación del repositorio:

```bash
chmod +x run.sh
./run.sh
