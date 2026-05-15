import subprocess
import cv2
import numpy as np
import time
import random
from dotenv import load_dotenv
import os

load_dotenv()

ADB = "adb"
DEVICE = os.getenv("DEVICE_ID")

if not DEVICE:
    raise ValueError("DEVICE_ID not set — copy .env.example to .env and add your device ID")



def adb(cmd):
    """Run any adb command"""
    return subprocess.run(
        [ADB, "-s", DEVICE] + cmd,
        capture_output=True
    )

def screenshot():
    # Execute with -p for PNG format
    result = adb(["exec-out", "screencap", "-p"])
    
    if not result.stdout:
        raise ConnectionError("ADB returned empty — check USB connection or Waydroid status")

    # PNG Magic Number: All valid PNGs start with these specific bytes
    PNG_MAGIC = b'\x89PNG\r\n\x1a\n'
    
    # Locate the true start of the image data
    start_index = result.stdout.find(PNG_MAGIC)
    
    if start_index == -1:
        # Diagnostic: If no PNG header is found, check if ADB sent an error message instead
        error_msg = result.stdout.decode(errors='ignore')[:100]
        raise ConnectionError(f"Valid PNG header not found. ADB sent: {error_msg}")
    
    # Extract only the valid PNG data
    clean_data = result.stdout[start_index:]
    
    img_array = np.frombuffer(clean_data, dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    
    if img is None:
        raise ConnectionError("OpenCV failed to decode the binary buffer into an image")
        
    return img

def tap(x, y):
    """Basic tap — used internally"""
    adb(["shell", "input", "tap", str(x), str(y)])
    print(f"  → tapped ({x}, {y})")

def human_tap(x, y, radius=15):
    """Tap with random offset + random delay to simulate human"""
    rx = x + random.randint(-radius, radius)
    ry = y + random.randint(-radius, radius)
    adb(["shell", "input", "tap", str(rx), str(ry)])
    print(f"  → tapped ({rx}, {ry}) [target ({x}, {y})]")
    time.sleep(random.uniform(0.1, 0.4))

def save_screenshot(filename="screenshots/screen.png"):
    """Save screenshot to disk for inspection"""
    img = screenshot()
    if img is not None:
        cv2.imwrite(filename, img)
        print(f"  → saved to {filename}")
    else:
        print("  → ERROR: screenshot returned nothing, check ADB connection")
    return img

def crop_template(x, y, w, h, filename):
    """
    Crop a region from current screen and save as template
    x, y = top-left corner
    w, h = width and height
    """
    img = screenshot()
    crop = img[y:y+h, x:x+w]
    cv2.imwrite(f"templates/{filename}", crop)
    print(f"  → saved template: templates/{filename}")
    return crop

def find_template(template_path, threshold=0.8):
    """
    Find a template image on the current screen.
    Returns (x, y) center coordinates if found, None if not found.
    """
    screen = screenshot()
    template = cv2.imread(template_path)

    if template is None:
        print(f"  → ERROR: template not found at {template_path}")
        return None

    # Do the matching
    result = cv2.matchTemplate(screen, template, cv2.TM_CCOEFF_NORMED)

    # Get the best match score and location
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

    print(f"  → match score: {max_val:.2f} (need >{threshold})")

    if max_val >= threshold:
        # max_loc is top-left corner, calculate center
        h, w = template.shape[:2]
        center_x = max_loc[0] + w // 2
        center_y = max_loc[1] + h // 2
        print(f"  → found at center: ({center_x}, {center_y})")
        return (center_x, center_y)
    else:
        print(f"  → not found on screen")
        return None

def find_and_tap(template_path, threshold=0.6):
    pos = find_template(template_path, threshold)
    if pos:
        time.sleep(random.uniform(0.1, 0.6))  # reaction time before tapping
        human_tap(pos[0], pos[1])
        return True
    return False

def wait_for_template(template_path, timeout=60, threshold=0.8):
    """Keep checking until template appears — with random poll interval"""
    print(f"  → waiting for {template_path}...")
    start = time.time()
    while time.time() - start < timeout:
        if find_template(template_path, threshold):
            return True
        time.sleep(random.uniform(1.5, 3.5))  # varied poll interval
    print(f"  → TIMEOUT waiting for {template_path}")
    return False

def human_sleep(min_s, max_s):
    """Random sleep between min and max seconds"""
    t = random.uniform(min_s, max_s)
    time.sleep(t)

def normalize_view():
    """Scroll to expose top-left deployment edge — with random variation"""
    scrolls_up = random.randint(2, 3)
    scrolls_left = random.randint(1, 3)

    for _ in range(scrolls_up):
        # Slightly randomize swipe start/end points
        sx = random.randint(650, 750)
        sy = random.randint(280, 320)
        ey = random.randint(520, 580)
        dur = random.randint(350, 500)
        adb(["shell", "input", "swipe", str(sx), str(sy), str(sx), str(ey), str(dur)])
        time.sleep(random.uniform(0.3, 0.6))

    for _ in range(scrolls_left):
        sx = random.randint(450, 550)
        sy = random.randint(320, 380)
        ex = random.randint(750, 850)
        dur = random.randint(350, 500)
        adb(["shell", "input", "swipe", str(sx), str(sy), str(ex), str(sy), str(dur)])
        time.sleep(random.uniform(0.3, 0.6))

    time.sleep(random.uniform(0.4, 0.8))


def find_red_zone_edge(scan_from_x=50, scan_to_x=800, scan_y_range=(50, 500)):
    """
    Detect the top-left edge of the red (no-deploy) zone.
    Scans columns from left to right, finds topmost red pixel in each column.
    Returns list of (x, y) points along the edge — just outside red zone.
    """
    screen = screenshot()
    edge_points = []

    for x in range(scan_from_x, scan_to_x, 25):
        for y in range(scan_y_range[0], scan_y_range[1]):
            b, g, r = screen[y][x]

            # Red zone detection — red channel dominant
            is_red_zone = (
                r > 150 and        # strong red
                r > g * 1.5 and    # much redder than green
                r > b * 1.5        # much redder than blue
            )

            if is_red_zone:
                # Step back 15px outside the red zone
                deploy_y = max(y - 15, 0)
                deploy_x = max(x - 15, 0)
                edge_points.append((deploy_x, deploy_y))
                break  # found topmost red in this column, move to next

    return edge_points

def deploy_clustered(edge_points, count=12, repeat=2):
    """
    Deploy troops in a cluster around the center of the edge.
    Drops each troop 'repeat' times to handle red zone misses.
    Adds human-like randomness to each tap.
    """
    if not edge_points:
        print("  → ERROR: no edge points")
        return

    # Focus on center 40% of the edge line for clustering
    center_start = len(edge_points) // 3
    center_end = 2 * len(edge_points) // 3
    cluster_points = edge_points[center_start:center_end]

    if not cluster_points:
        cluster_points = edge_points  # fallback to all points

    print(f"  → deploying in cluster of {len(cluster_points)} points")

    taps_done = 0
    target_taps = count * repeat  # e.g. 12 troops * 2 = 24 taps

    while taps_done < target_taps:
        # Pick a random point from cluster each time
        x, y = random.choice(cluster_points)
        human_tap(x, y)
        taps_done += 1
        # Random pause between troops like a human would
        time.sleep(random.uniform(0.15, 0.5))

    print(f"  → done ({taps_done} taps for {count} troops)")

def deploy_hero(icon_template, edge_points, repeats=3):
    """
    Select hero and tap deployment spot multiple times.
    Retries icon tap if not found.
    """
    # Try tapping icon up to 3 times
    for attempt in range(3):
        if find_and_tap(icon_template):
            break
        print(f"  → hero icon not found, retrying ({attempt+1}/3)...")
        time.sleep(0.5)

    time.sleep(random.uniform(0.3, 0.6))

    # Pick a point from center cluster, tap it multiple times
    center_start = len(edge_points) // 3
    center_end = 2 * len(edge_points) // 3
    cluster = edge_points[center_start:center_end]
    if not cluster:
        cluster = edge_points

    for _ in range(repeats):
        x, y = random.choice(cluster)
        human_tap(x, y, radius=20)
        time.sleep(random.uniform(0.2, 0.5))

def use_hero_abilities(edge_points):
    """
    Use hero abilities mid-battle by tapping hero icons again.
    Called ~30s after deployment when heroes are inside the base.
    """
    print("[Abilities] Activating hero abilities...")
    
    # Each hero ability is activated by tapping their portrait
    # which appears on screen during battle (bottom left area)
    for template in ["templates/king_icon.png", 
                     "templates/queen_icon.png",
                     "templates/warden_icon.png"]:
        result = find_and_tap(template, threshold=0.2)
        if result:
            print(f"  → ability used")
        else:
            print(f"  → hero not found/available")
        time.sleep(random.uniform(0.5, 1.0))

def visualize_edge(edge_points):
    """Draw detected edge points on screenshot for verification"""
    screen = screenshot()
    for (x, y) in edge_points:
        cv2.circle(screen, (x, y), 8, (255, 0, 0), -1)  # blue dots
    cv2.imwrite("screenshots/edge_debug.png", screen)
    print(f"  → found {len(edge_points)} edge points")

# --- CONFIGURATION ---
# Map filenames to keys (ensure these are in your /templates folder)
TROOP_TEMPLATES = {
    "e_dragon": "templates/e_dragon.png",
    "balloon": "templates/baloon.png",
    "minion_prince": "templates/minion_prince.png",
    "queen": "templates/queen.png",
    "warden": "templates/warden.png",
    "royal_champion": "templates/royal_champion.png",
    "stone_slammer": "templates/stone_slamer.png"
}

# The order in which units are selected and dropped
# Heroes first to tank, followed by main air force
DEPLOYMENT_ORDER = [
    "queen", 
    "warden", 
    "royal_champion", 
    "stone_slammer",
    "e_dragon", 
    "balloon", 
    "minion_prince"
]

def run_attack():
    """Refactored attack cycle using new templates"""

    # --- STATE 1-3: Navigation ---
    print("\n[State 1-3] Navigating to battle...")
    if not find_and_tap("templates/attack_btn.png"): return False
    human_sleep(1.5, 3.0)
    
    if not find_and_tap("templates/find_match_btn.png"): return False
    human_sleep(1.5, 3.0)
    
    if not find_and_tap("templates/confirm_attack_btn.png"): return False
    human_sleep(6, 10) 

    # --- STATE 4: Deployment ---
    print("[State 4] Normalizing view and detecting edge...")
    normalize_view()
    edge_points = find_red_zone_edge()

    if not edge_points:
        print("  → No edge detected, looking for Next button...")
        find_and_tap("templates/next_btn.png") # Uses your next_btn.png or next_text.png
        return False

    visualize_edge(edge_points)
    
    for unit in DEPLOYMENT_ORDER:
        template_path = TROOP_TEMPLATES[unit]
        print(f"[Deploy] Selecting {unit}...")
        
        if find_and_tap(template_path, threshold=0.75):
            human_sleep(0.2, 0.5)
            
            # Use specific drop logic
            if unit in ["queen", "warden", "royal_champion", "stone_slammer"]:
                # Single units get focused drops
                deploy_hero(template_path, edge_points, repeats=2)
            else:
                # Mass troops get clustered drops
                deploy_clustered(edge_points, count=12, repeat=1)
            
            human_sleep(0.3, 0.7)
        else:
            print(f"  → {unit} icon not found, skipping.")

    # --- STATE 5-6: Battle Management ---
    print("[State 5] Waiting for battle to finish...")
    # Abilities check after 25 seconds
    time.sleep(25) 
    use_hero_abilities(edge_points)

    if not wait_for_template("templates/return_home_btn.png", timeout=210):
        print("  → Battle timed out")
        return False

    print("[State 6] Returning Home...")
    find_and_tap("templates/return_home_btn.png")
    human_sleep(4, 7)
    return True

def idle_behavior():
    """
    Occasionally do nothing or random scroll on home screen.
    Makes the bot look like a human who pauses to think.
    """
    behavior = random.choice(["pause", "scroll", "double_pause"])
    
    if behavior == "pause":
        t = random.uniform(15, 40)
        print(f"  → idling for {t:.0f}s")
        time.sleep(t)
    
    elif behavior == "scroll":
        t = random.uniform(8, 20)
        print(f"  → idle scrolling for {t:.0f}s")
        time.sleep(t)
        adb(["shell", "input", "swipe",
             str(random.randint(400, 800)),
             str(random.randint(200, 400)),
             str(random.randint(400, 800)),
             str(random.randint(200, 400)),
             str(random.randint(300, 700))])
        time.sleep(random.uniform(3, 8))  # pause after scroll
    
    elif behavior == "double_pause":
        time.sleep(random.uniform(20, 40))

# --- MAIN LOOP ---
if __name__ == "__main__":
    attack_count = 0
    max_attacks = random.randint(8, 12)  # vary the session length too
    session_start = time.time()
    max_session_hours = random.uniform(1.5, 3.0)  # bot for 1.5-3hrs then stop

    print("=== COC BOT STARTING ===")
    print(f"Session limit: {max_session_hours:.1f} hours, max {max_attacks} attacks\n")

    while attack_count < max_attacks:
        # Check session time limit
        elapsed_hours = (time.time() - session_start) / 3600
        if elapsed_hours >= max_session_hours:
            print(f"\n[Session] Time limit reached ({elapsed_hours:.1f}h), stopping.")
            break

        attack_count += 1
        print(f"\n{'='*40}")
        print(f"ATTACK #{attack_count} | Session: {elapsed_hours:.1f}h/{max_session_hours:.1f}h")
        print(f"{'='*40}")

        success = run_attack()

        if success:
            print(f"  ✓ Attack #{attack_count} completed!")
        else:
            print(f"  ✗ Attack #{attack_count} had issues")
            human_sleep(3, 8)

        idle_behavior()

    print(f"\n=== BOT FINISHED === {attack_count} attacks in {elapsed_hours:.1f}h") 




