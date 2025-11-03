from flask import Blueprint, Response, render_template, jsonify

# Creamos un Blueprint. El primer argumento es el nombre del Blueprint,
# y el segundo es el nombre del módulo o paquete donde se encuentra,
# lo cual es necesario para que Flask encuentre las plantillas (templates).
api_bp = Blueprint('api', __name__, template_folder='../templates')

# Estas variables serán inyectadas desde web_streamer.py
generate_frames_func = None
get_vehicle_counter_func = None

def register_streaming_logic(generate_frames, get_vehicle_counter):
    """Función para inyectar la lógica de streaming desde el módulo principal."""
    global generate_frames_func, get_vehicle_counter_func
    generate_frames_func = generate_frames
    get_vehicle_counter_func = get_vehicle_counter

@api_bp.route("/")
def index():
    """Página principal que muestra el video."""
    # Nota: Flask buscará 'index.html' en la carpeta 'templates' relativa a donde se define el Blueprint.
    return render_template("index.html")

@api_bp.route("/video_feed")
def video_feed():
    """Ruta que sirve el stream de video."""
    if not generate_frames_func:
        return "Error: La lógica de streaming no está inicializada.", 500
    return Response(generate_frames_func(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

@api_bp.route("/api/count")
def count():
    """Endpoint de API que devuelve el conteo actual en formato JSON."""
    if not get_vehicle_counter_func:
        return jsonify({"error": "El contador no está inicializado"}), 500
    
    return jsonify({"vehicle_count": get_vehicle_counter_func()})