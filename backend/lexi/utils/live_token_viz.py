# lexi/live_token_viz.py
import socket, struct

print("[LEX] live_token_viz imported")

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
TARGET = ("127.0.0.1", 5005)


def send_to_viz(vec, z=0.0, intensity=1.0):
    try:
        if vec is None:
            print("[LEX][VIZ] skip: vec=None")
            return
        if hasattr(vec, "__len__") and len(vec) < 2:
            print(f"[LEX][VIZ] skip: len<2 ({len(vec)})")
            return
        # Accept list / numpy / torch
        if not isinstance(vec, (list, tuple)):
            try:
                vec = vec.tolist()
            except Exception:
                pass
        x = float(vec[0])
        y = float(vec[1])
        pkt = struct.pack("ffff", x, y, float(z), float(intensity))
        sock.sendto(pkt, TARGET)
        print(f"[LEX][VIZ] SENT ({x:.3f},{y:.3f},{z:.3f},{intensity:.3f})")
    except Exception as e:
        print("[LEX][VIZ] send error:", e)
