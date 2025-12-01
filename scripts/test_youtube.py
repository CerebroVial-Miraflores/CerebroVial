#!/usr/bin/env python3
"""
Test YouTube stream fluidity with different approaches.
"""
import cv2
import time
import sys

def test_with_streamlink(youtube_url: str):
    """Test YouTube stream using Streamlink."""
    print(f"\n{'='*60}")
    print("TEST 1: Streamlink approach")
    print(f"{'='*60}")
    print(f"URL: {youtube_url}")
    
    try:
        import streamlink
        print("Resolviendo stream con Streamlink...")
        streams = streamlink.streams(youtube_url)
        
        if not streams:
            print("❌ Streamlink no encontró streams disponibles.")
            return False
            
        if "best" in streams:
            real_url = streams["best"].url
            print(f"✅ Streamlink URL resuelta: {real_url[:80]}...")
            
            cap = cv2.VideoCapture(real_url)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 60)
            
            if not cap.isOpened():
                print("❌ Error: No se pudo abrir el stream.")
                return False
            
            print("Reproduciendo durante 30 segundos...")
            print("Presiona 'q' para salir antes.")
            
            frame_count = 0
            start_time = time.time()
            fps_start_time = time.time()
            fps_frame_count = 0
            test_duration = 30
            
            while True:
                read_start = time.time()
                ret, frame = cap.read()
                read_duration = time.time() - read_start
                
                if read_duration > 0.1:
                    print(f"⚠️  Lag detectado: Lectura tardó {read_duration:.2f}s")
                
                if not ret:
                    print("⚠️  No se pudo leer el frame.")
                    break
                
                frame_count += 1
                fps_frame_count += 1
                
                # FPS cada segundo
                if time.time() - fps_start_time >= 1.0:
                    fps = fps_frame_count / (time.time() - fps_start_time)
                    print(f"FPS: {fps:.2f}")
                    fps_frame_count = 0
                    fps_start_time = time.time()
                
                cv2.imshow("Streamlink Test", frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
                if time.time() - start_time > test_duration:
                    print(f"\n✅ Test completado: {test_duration}s")
                    break
            
            cap.release()
            cv2.destroyAllWindows()
            
            total_time = time.time() - start_time
            avg_fps = frame_count / total_time
            print(f"\nEstadísticas:")
            print(f"  Tiempo total: {total_time:.2f}s")
            print(f"  Frames totales: {frame_count}")
            print(f"  FPS promedio: {avg_fps:.2f}")
            
            return True
            
    except Exception as e:
        print(f"❌ Error con Streamlink: {e}")
        return False

def test_with_ytdlp(youtube_url: str):
    """Test YouTube stream using yt-dlp."""
    print(f"\n{'='*60}")
    print("TEST 2: yt-dlp approach")
    print(f"{'='*60}")
    print(f"URL: {youtube_url}")
    
    try:
        import yt_dlp
        
        ydl_opts = {
            'format': 'best',
            'noplaylist': True,
            'quiet': True,
            'extractor_args': {'youtube': {'player_client': ['default']}}
        }
        
        print("Extrayendo URL con yt-dlp...")
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(youtube_url, download=False)
            stream_url = info['url']
            print(f"✅ URL extraída: {stream_url[:80]}...")
            
            cap = cv2.VideoCapture(stream_url)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 60)
            
            if not cap.isOpened():
                print("❌ Error: No se pudo abrir el stream.")
                return False
            
            print("Reproduciendo durante 30 segundos...")
            print("Presiona 'q' para salir antes.")
            
            frame_count = 0
            start_time = time.time()
            fps_start_time = time.time()
            fps_frame_count = 0
            test_duration = 30
            
            while True:
                read_start = time.time()
                ret, frame = cap.read()
                read_duration = time.time() - read_start
                
                if read_duration > 0.1:
                    print(f"⚠️  Lag detectado: Lectura tardó {read_duration:.2f}s")
                
                if not ret:
                    print("⚠️  No se pudo leer el frame.")
                    break
                
                frame_count += 1
                fps_frame_count += 1
                
                # FPS cada segundo
                if time.time() - fps_start_time >= 1.0:
                    fps = fps_frame_count / (time.time() - fps_start_time)
                    print(f"FPS: {fps:.2f}")
                    fps_frame_count = 0
                    fps_start_time = time.time()
                
                cv2.imshow("yt-dlp Test", frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
                if time.time() - start_time > test_duration:
                    print(f"\n✅ Test completado: {test_duration}s")
                    break
            
            cap.release()
            cv2.destroyAllWindows()
            
            total_time = time.time() - start_time
            avg_fps = frame_count / total_time
            print(f"\nEstadísticas:")
            print(f"  Tiempo total: {total_time:.2f}s")
            print(f"  Frames totales: {frame_count}")
            print(f"  FPS promedio: {avg_fps:.2f}")
            
            return True
            
    except Exception as e:
        print(f"❌ Error con yt-dlp: {e}")
        return False

def main():
    # URL de YouTube para probar (Lofi Girl - stream 24/7)
    youtube_url = "https://www.youtube.com/watch?v=jfKfPfyJRdk"
    
    if len(sys.argv) > 1:
        youtube_url = sys.argv[1]
    
    print("\n" + "="*60)
    print("PRUEBA DE FLUIDEZ DE YOUTUBE")
    print("="*60)
    
    # Test 1: Streamlink
    streamlink_ok = test_with_streamlink(youtube_url)
    
    # Test 2: yt-dlp
    ytdlp_ok = test_with_ytdlp(youtube_url)
    
    # Resumen
    print(f"\n{'='*60}")
    print("RESUMEN")
    print(f"{'='*60}")
    print(f"Streamlink: {'✅ OK' if streamlink_ok else '❌ FALLÓ'}")
    print(f"yt-dlp:     {'✅ OK' if ytdlp_ok else '❌ FALLÓ'}")
    print()

if __name__ == "__main__":
    main()
