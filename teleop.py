
import matplotlib.pyplot as plt

# ---------- Teleoperation Controller ----------
class TeleoperationController:
    """
    Simple keyboard teleoperation controller.
    Arrow keys: control robot movement
    Space: stop/pause
    """
    def __init__(self, default_speed=2.0, default_turn=5.0):
        self.dx = 0.0
        self.dtheta = 0.0
        self.default_speed = default_speed
        self.default_turn = default_turn
        self.paused = False
        self.mode = 'teleop'  # 'teleop' or 'auto'
        
        print("\n=== TELEOPERATION CONTROLS ===")
        print("Arrow Up    : Move forward")
        print("Arrow Down  : Move backward")
        print("Arrow Left  : Turn left")
        print("Arrow Right : Turn right")
        print("Space       : Stop/Pause")
        print("'a'         : Toggle Auto/Manual mode")
        print("'q'         : Quit")
        print("'+'         : Increase speed")
        print("'-'         : Decrease speed")
        print("==============================\n")
    
    def on_key_press(self, event):
        """Handle key press events"""
        if event.key == 'up':
            self.dx = self.default_speed
            self.dtheta = 0.0
            self.paused = False
            print(f"↑ Forward: dx={self.dx:.1f}")
        
        elif event.key == 'down':
            self.dx = -self.default_speed * 0.5  # slower backward
            self.dtheta = 0.0
            self.paused = False
            print(f"↓ Backward: dx={self.dx:.1f}")
        
        elif event.key == 'left':
            self.dx = self.default_speed * 0.7  # slower when turning
            self.dtheta = -self.default_turn
            self.paused = False
            print(f"← Turn Left: dθ={self.dtheta:.1f}°")
        
        elif event.key == 'right':
            self.dx = self.default_speed * 0.7
            self.dtheta = self.default_turn
            self.paused = False
            print(f"→ Turn Right: dθ={self.dtheta:.1f}°")
        
        elif event.key == ' ':  # spacebar
            self.paused = not self.paused
            if self.paused:
                self.dx = 0.0
                self.dtheta = 0.0
                print("⏸ PAUSED")
            else:
                print("▶ RESUMED")
        
        elif event.key == 'a':
            self.mode = 'auto' if self.mode == 'teleop' else 'teleop'
            print(f"Mode switched to: {self.mode.upper()}")
        
        elif event.key == '+' or event.key == '=':
            self.default_speed += 0.5
            self.default_turn += 1.0
            print(f"Speed increased: {self.default_speed:.1f} px, {self.default_turn:.1f}°")
        
        elif event.key == '-' or event.key == '_':
            self.default_speed = max(0.5, self.default_speed - 0.5)
            self.default_turn = max(1.0, self.default_turn - 1.0)
            print(f"Speed decreased: {self.default_speed:.1f} px, {self.default_turn:.1f}°")
        
        elif event.key == 'q':
            print("Quitting...")
            plt.close('all')
    
    def on_key_release(self, event):
        """Handle key release events - stop motion when key released"""
        if event.key in ['up', 'down', 'left', 'right']:
            if not self.paused:
                self.dx = 0.0
                self.dtheta = 0.0
    
    def get_command(self):
        """Get current control command"""
        if self.paused:
            return 0.0, 0.0
        return self.dx, self.dtheta
