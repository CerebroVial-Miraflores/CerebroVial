"""
scripts/test_stream.py

Script de diagnóstico para probar la fluidez del stream de video crudo.
Sin procesamiento de IA, solo lectura y visualización con OpenCV.
"""
import cv2
import time
import sys

# URL de la cámara de Javier Prado
DEFAULT_SOURCE = "https://live.smartechlatam.online/claro/javierprado/index.m3u8"

def main():
    source_url = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_SOURCE
    
    print(f"Abriendo stream: {source_url}")
    print("Presiona 'q' para salir.")
    
    # Resolver URL con Streamlink
    try:
        import streamlink
        print("Resolviendo stream con Streamlink...")
        streams = streamlink.streams(source_url)
        if "best" in streams:
            real_url = streams["best"].url
            print(f"✅ Streamlink URL resuelta: {real_url[:50]}...")
            cap = cv2.VideoCapture(real_url)
        else:
            print("⚠️  Streamlink no encontró stream 'best', usando URL original.")
            cap = cv2.VideoCapture(source_url)
    except Exception as e:
        print(f"⚠️  Error con Streamlink: {e}. Usando URL original.")
        cap = cv2.VideoCapture(source_url)
    
    if not cap.isOpened():
        print("❌ Error: No se pudo abrir el stream.")
        return

    # Aumentar buffer para estabilidad (aprox 2 segundos a 30 FPS)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 60)
    
    frame_count = 0
    start_time = time.time()
    fps_start_time = time.time()
    fps_frame_count = 0
    
    try:
        while True:
            read_start = time.time()
            ret, frame = cap.read()
            read_duration = time.time() - read_start
            
            if read_duration > 0.1:
                print(f"⚠️  Lag de red detectado: Lectura tardó {read_duration:.2f}s")
            
            if not ret:
                print("⚠️  No se pudo leer el frame (stream cortado o finalizado).")
                # Reintentar conexión si es un stream en vivo
                time.sleep(1)
                cap.release()
                cap = cv2.VideoCapture(source_url)
                continue
            
            # Redimensionar para visualización consistente
            frame = cv2.resize(frame, (1280, 720))
            
            # Calcular FPS de visualización
            fps_frame_count += 1
            if time.time() - fps_start_time > 1.0:
                fps = fps_frame_count / (time.time() - fps_start_time)
                print(f"FPS: {fps:.2f}")
                fps_frame_count = 0
                fps_start_time = time.time()
            
            cv2.imshow("Test Stream (Raw)", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
            frame_count += 1
            
    except KeyboardInterrupt:
        print("\nInterrumpido por usuario.")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("✅ Test finalizado.")

if __name__ == "__main__":
    main()
